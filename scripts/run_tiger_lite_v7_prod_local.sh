#!/bin/bash
# tiger_lite_v7_prod 로컬 2-stage 학습 러너
#   stage 1: SID Tokenizer v3 (configs/sid_v2_v3.yaml)
#       산출: /home/jeongmindo/projects/ably/data/reco_experiments/v2_full/meta_v3/item_to_sid.parquet
#   stage 2: TIGER-lite v7 (configs/tiger_lite_v7.yaml) — stage 1 산출 사용
#       ckpt: logs/tiger_lite/v7/fit/checkpoints/best_tiger_lite_v7*.ckpt
#
# 사용:
#   nohup bash scripts/run_tiger_lite_v7_prod_local.sh > $LOG 2>&1 &
set -euo pipefail

REPO=/home/jeongmindo/projects/ably/reco-lightning
DATA_LOG_DIR=/home/jeongmindo/projects/ably/data/reco_experiments/v2_full/logs
mkdir -p "$DATA_LOG_DIR"

cd "$REPO"
PY="$REPO/.venv/bin/python"

stage() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [tiger_lite_v7_prod] $*"; }

stage "=== Stage 1/2: SID Tokenizer v3 (sid_v2_v3.yaml) ==="
"$PY" src/main.py fit -c configs/sid_v2_v3.yaml
stage "stage1 완료"

# Stage1 산출 확인
SID_OUT=/home/jeongmindo/projects/ably/data/reco_experiments/v2_full/meta_v3/item_to_sid.parquet
if [ ! -f "$SID_OUT" ]; then
    stage "ERROR: stage1 산출 없음: $SID_OUT"
    exit 1
fi
stage "stage1 산출 확인: $SID_OUT ($(du -h "$SID_OUT" | cut -f1))"

stage "=== Stage 2/2: TIGER-lite v7 (tiger_lite_v7.yaml) ==="
"$PY" src/main.py fit -c configs/tiger_lite_v7.yaml
stage "stage2 완료"

stage "=== Validate (best ckpt) ==="
BEST_CKPT=$(ls -t "$REPO"/logs/tiger_lite/v7/fit/checkpoints/best_tiger_lite_v7*.ckpt 2>/dev/null | head -1 || true)
if [ -n "$BEST_CKPT" ]; then
    "$PY" src/main.py validate -c configs/tiger_lite_v7.yaml --ckpt_path "$BEST_CKPT"
    stage "validate 완료: $BEST_CKPT"
else
    stage "WARN: best ckpt 못찾음"
fi

stage "=== ALL DONE ==="
