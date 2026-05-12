#!/bin/bash
# 멀티-GPU runpod pod 내부에서 여러 실험을 GPU 0..N-1 에 fan-out 해서 병렬 실행.
#
# 사용법:
#   MODELS=cbf3-e0,cbf3-e1,cbf3-e2,cbf3-e3,cbf3-e4,cbf3-e5,cbf3-e6,cbf3-e7 \
#       ./scripts/runpod_pod_run_all.sh
#
# 첫 모델만 동기로 데이터 다운로드 → 나머지는 SKIP_DOWNLOAD=1 로 병렬 fan-out.
# 각 실험의 stdout 은 /workspace/logs/<MODEL>.stdout 으로 분리 저장.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODELS:?MODELS required (콤마 구분, 최대 8개, 예: cbf3-e0,cbf3-e1,...)}"
export DATE="${DATE:-20260506}"
export DATA_ROOT="${DATA_ROOT:-/data}"
export WANDB_PROJECT="${WANDB_PROJECT:-cbf-exp}"

log() { echo "[pod_run_all] $*"; }

IFS=',' read -ra ARR <<< "$MODELS"
N=${#ARR[@]}
if [ "$N" -gt 8 ] || [ "$N" -lt 1 ]; then
    log "ERROR: MODELS 개수는 1~8 (현재: $N)"
    exit 1
fi
log "Models ($N): ${ARR[*]}"

mkdir -p /workspace/logs

# ---- 1) 데이터 동기 다운로드 (첫 1회만) ----
# 8개 동시에 aws s3 sync 하면 같은 파일에 race → corruption 가능.
# 첫 모델 download 후 marker 생성, 이후 fan-out 들은 SKIP_DOWNLOAD=1.
MARKER="$DATA_ROOT/.cbf3_pod_download_done"
if [ ! -f "$MARKER" ]; then
    log "[1/2] 첫 실행 — synchronous download via MODEL=${ARR[0]}"
    GPU_ID=0 MODEL="${ARR[0]}" \
        SKIP_DOWNLOAD=0 \
        KEEP_LOGS=1 KEEP_MODEL_PATH=1 \
        TERMINATE_ON_EXIT=0 \
        SKIP_UV_SYNC=1 \
        DOWNLOAD_ONLY=1 \
        bash -c '
            set -e
            cd /workspace
            # entrypoint 의 download 단계까지만 돌리고 학습은 skip — 마커 생성 후 종료
            bash "'"$SCRIPT_DIR"'/runpod_download.sh" "$MODEL" "$DATE" "$DATA_ROOT"
            touch "'"$MARKER"'"
        '
    log "    download 완료 → $MARKER"
else
    log "[1/2] cache 발견 ($MARKER) — download 건너뜀"
fi

# ---- 2) 병렬 fan-out: GPU i ← ARR[i] ----
log "[2/2] $N 개 실험 병렬 시작"
PIDS=()
NAMES=()
for i in "${!ARR[@]}"; do
    M="${ARR[$i]}"
    LOG_FILE="/workspace/logs/${M}.stdout"
    log "  GPU $i ← $M  (log: $LOG_FILE)"
    (
        GPU_ID="$i" MODEL="$M" \
            "$SCRIPT_DIR/runpod_pod_run_one.sh"
    ) > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    NAMES+=("$M")
done

# ---- 3) Wait + 결과 집계 ----
log "waiting on ${#PIDS[@]} PIDs..."
FAILED=()
for j in "${!PIDS[@]}"; do
    pid="${PIDS[$j]}"
    name="${NAMES[$j]}"
    if wait "$pid"; then
        log "  [OK]   $name (PID $pid)"
    else
        rc=$?
        log "  [FAIL] $name (PID $pid, exit=$rc) — see /workspace/logs/${name}.stdout"
        FAILED+=("$name")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    log "DONE — ${#FAILED[@]} 개 실패: ${FAILED[*]}"
    exit 1
fi
log "DONE — 전부 성공"
