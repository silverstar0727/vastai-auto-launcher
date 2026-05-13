#!/bin/bash
# 로컬 호스트에서 도커 없이 여러 실험을 GPU 0..N-1 에 fan-out 해서 병렬 실행.
# (runpod_pod_run_all.sh 의 host 버전)
#
# 사용법:
#   MODELS=cbf3-e0,cbf3-e1,cbf3-e2,cbf3-e3 ./scripts/local_run_all.sh
#
# 첫 모델만 동기 download → 나머지는 SKIP_DOWNLOAD=1 로 병렬.
# 각 실험 stdout 은 logs/<MODEL>.stdout 로 분리.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

: "${MODELS:?MODELS required (콤마 구분, 예: cbf3-e0,cbf3-e1,...)}"
export DATE="${DATE:-20260506}"
export DATA_ROOT="${DATA_ROOT:-$HOME/projects/ably/data}"
export AWS_PROFILE="${AWS_PROFILE:-MLEngineer-145481888492}"

log() { echo "[local_run_all] $*"; }

IFS=',' read -ra ARR <<< "$MODELS"
N=${#ARR[@]}

# GPU 개수 확인 (CUDA_VISIBLE_DEVICES 가 설정돼있으면 그 개수, 없으면 nvidia-smi 결과)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

if [ "$N" -gt "$NUM_GPU" ]; then
    log "ERROR: 실험 $N 개인데 GPU $NUM_GPU 개 — fan-out 불가"
    exit 1
fi
log "Models ($N): ${ARR[*]} / GPU $NUM_GPU 사용"

cd "$PROJECT_DIR"
mkdir -p logs

# ---- 1) Download 동기 (첫 1회만) ----
MARKER="$DATA_ROOT/.local_run_download_done"
if [ ! -f "$MARKER" ]; then
    log "[1/2] 첫 실행 — synchronous SSO + download"
    # 첫 모델 download 만 시키고 학습 진입은 막아야 함 → runpod_download.sh 직접 호출
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
        log "    SSO 만료/미인증 — 자동 로그인"
        aws sso login --profile "$AWS_PROFILE"
    fi
    bash "$SCRIPT_DIR/runpod_download.sh" "${ARR[0]}" "$DATE" "$DATA_ROOT"
    touch "$MARKER"
    log "    download 완료 → $MARKER"
else
    log "[1/2] cache 발견 ($MARKER) — download 건너뜀 (지우면 재다운로드)"
fi

# ---- 2) 병렬 fan-out ----
log "[2/2] $N 개 실험 병렬 시작"
PIDS=()
NAMES=()
for i in "${!ARR[@]}"; do
    M="${ARR[$i]}"
    LOG_FILE="$PROJECT_DIR/logs/${M}.stdout"
    log "  GPU $i ← $M  (log: $LOG_FILE)"
    (
        GPU_ID="$i" MODEL="$M" \
            SKIP_DOWNLOAD=1 \
            KEEP_LOGS=1 KEEP_MODEL_PATH=1 \
            "$SCRIPT_DIR/local_run.sh"
    ) > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    NAMES+=("$M")
done

# ---- 3) Wait + 집계 ----
log "waiting on ${#PIDS[@]} PIDs..."
FAILED=()
for j in "${!PIDS[@]}"; do
    pid="${PIDS[$j]}"
    name="${NAMES[$j]}"
    if wait "$pid"; then
        log "  [OK]   $name (PID $pid)"
    else
        rc=$?
        log "  [FAIL] $name (PID $pid, exit=$rc) — see logs/${name}.stdout"
        FAILED+=("$name")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    log "DONE — ${#FAILED[@]} 개 실패: ${FAILED[*]}"
    exit 1
fi
log "DONE — 전부 성공"
