#!/bin/bash
set -e

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/output/multi_interest/preprocessed/
rm -rf logs/multi-interest/

echo "===== Multi-Interest Training ====="
uv run python -u src/main.py fit -c configs/multi_interest.yaml

echo "===== Done ====="
