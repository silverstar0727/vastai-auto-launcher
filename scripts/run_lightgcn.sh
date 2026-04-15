#!/bin/bash
set -e

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/lightgcn/input/data/train_data/
rm -rf /home/jeongmindo/projects/ably/data/output/lightgcn/
rm -rf logs/lightgcn/

mkdir -p /home/jeongmindo/projects/ably/data/output/lightgcn

echo "===== LightGCN Training ====="
uv run python -u src/main.py fit -c configs/lightgcn.yaml

echo "===== Done ====="
