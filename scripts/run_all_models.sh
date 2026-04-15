#!/bin/bash
set -e

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/output/cbf/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/output/complement/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/output/multi_interest/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/lightgcn/input/data/train_data/
rm -rf /home/jeongmindo/projects/ably/data/output/lightgcn/
rm -rf logs/cbf-cosine-warmup/ logs/complement-cosine-warmup/ logs/multi-interest/ logs/lightgcn/

mkdir -p /home/jeongmindo/projects/ably/data/output/lightgcn

echo "===== [1/4] CBF Training ====="
uv run python -u src/main.py fit -c configs/cbf.yaml
echo "===== CBF Done ====="

echo "===== [2/4] Complement Training ====="
uv run python -u src/main.py fit -c configs/complement.yaml
echo "===== Complement Done ====="

echo "===== [3/4] Multi-Interest Training ====="
uv run python -u src/main.py fit -c configs/multi_interest.yaml
echo "===== Multi-Interest Done ====="

echo "===== [4/4] LightGCN Training ====="
uv run python -u src/main.py fit -c configs/lightgcn.yaml
echo "===== LightGCN Done ====="

echo "===== All 4 Models Done ====="
