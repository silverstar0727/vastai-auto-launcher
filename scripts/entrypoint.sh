#!/bin/bash
set -e

echo "=========================================="
echo "Starting training pipeline..."
echo "=========================================="

# 1. S3에서 데이터 다운로드
if [ -n "$S3_DATA_PATH" ]; then
    echo "[1/3] Downloading data from S3: $S3_DATA_PATH"
    mkdir -p /workspace/data
    aws s3 sync "$S3_DATA_PATH" /workspace/data/ --quiet
    echo "Data download complete!"
else
    echo "[1/3] S3_DATA_PATH not set, skipping data download"
fi

# 2. W&B 로그인
if [ -n "$WANDB_API_KEY" ]; then
    echo "[2/3] Logging into W&B..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login complete!"
else
    echo "[2/3] WANDB_API_KEY not set, skipping W&B login"
fi

# 3. 학습 실행
CONFIG_FILE="${CONFIG_FILE:-configs/config.yaml}"
echo "[3/3] Starting training..."
echo "Command: python3 -u src/main.py fit -c $CONFIG_FILE $@"
echo "=========================================="

python3 -u src/main.py fit -c "$CONFIG_FILE" "$@"

TRAIN_EXIT_CODE=$?
echo "=========================================="
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

# 4. 학습 완료 후 Vast.ai 인스턴스 종료
if [ -n "$CONTAINER_ID" ] && [ -n "$VAST_API_KEY" ]; then
    echo "Destroying Vast.ai instance: $CONTAINER_ID"
    vastai destroy instance "$CONTAINER_ID" --api-key "$VAST_API_KEY"
    echo "Instance destroyed!"
else
    echo "CONTAINER_ID or VAST_API_KEY not set, skipping instance destruction"
fi

exit $TRAIN_EXIT_CODE
