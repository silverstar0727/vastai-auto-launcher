#!/bin/bash
# 멀티-GPU runpod pod 내부에서 "1개 실험"을 특정 GPU 에 핀해서 띄우는 wrapper.
#
# 사용법:
#   GPU_ID=0 MODEL=cbf3-e0 ./scripts/runpod_pod_run_one.sh    # 첫 실행 (download 수행)
#   GPU_ID=1 MODEL=cbf3-e1 ./scripts/runpod_pod_run_one.sh    # 이후 실행 (cache 사용)
#
# 첫 실행에서만 다운로드/AWS 인증을 거치고, 이후 실행은 SKIP_DOWNLOAD=1 강제.
# 동시 실행 안전성을 위해 KEEP_LOGS=1 / KEEP_MODEL_PATH=1 / TERMINATE_ON_EXIT=0 default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GPU_ID:?GPU_ID required (0 ~ 7)}"
: "${MODEL:?MODEL required (예: cbf3-e0)}"
export DATE="${DATE:-20260506}"
export DATA_ROOT="${DATA_ROOT:-/data}"
export WANDB_PROJECT="${WANDB_PROJECT:-cbf-exp}"

# Pod 내부 동시 실행 safety defaults — 이미 download 됐다고 가정.
# 첫 실험만 SKIP_DOWNLOAD=0 으로 override 해서 호출하면 됨.
export SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
export KEEP_LOGS="${KEEP_LOGS:-1}"
export KEEP_MODEL_PATH="${KEEP_MODEL_PATH:-1}"
export TERMINATE_ON_EXIT="${TERMINATE_ON_EXIT:-0}"

# uv sync 는 컨테이너 첫 부팅에서 끝났다고 가정 — 매 실험마다 다시 안 함
export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"

echo "[pod_run_one] GPU=$GPU_ID MODEL=$MODEL DATE=$DATE"
echo "[pod_run_one] cache flags: SKIP_DOWNLOAD=$SKIP_DOWNLOAD KEEP_LOGS=$KEEP_LOGS KEEP_MODEL_PATH=$KEEP_MODEL_PATH"

exec "$SCRIPT_DIR/runpod_entrypoint.sh"
