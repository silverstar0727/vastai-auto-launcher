#!/bin/bash
set -e

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/output/multi_interest/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/lightgcn/input/data/train_data/
rm -rf /home/jeongmindo/projects/ably/data/output/lightgcn/
rm -rf logs/multi_interest/ logs/lightgcn/

mkdir -p /home/jeongmindo/projects/ably/data/output/lightgcn

echo "===== [1/2] Multi-Interest Training ====="
uv run python -u src/main.py fit -c configs/multi_interest.yaml
echo "===== Multi-Interest Done ====="

echo "===== [2/2] LightGCN Training ====="
uv run python -u src/main.py fit -c configs/lightgcn.yaml
echo "===== LightGCN Done ====="

echo "===== All Done ====="
