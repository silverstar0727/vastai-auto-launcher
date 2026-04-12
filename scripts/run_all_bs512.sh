#!/bin/bash
set -e

echo "===== Sleeping 6 hours to wait for current run ====="
sleep 6h

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/output/cbf/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/output/complement/preprocessed/
rm -rf logs/cbf-cosine-warmup-bs512/ logs/complement-cosine-warmup-bs512/

echo "===== [1/2] CBF Training (bs512) ====="
uv run python src/main.py fit -c configs/cbf-cosine-warmup-bs512.yaml
echo "===== CBF Done ====="

echo "===== [2/2] Complement Training (bs512) ====="
uv run python src/main.py fit -c configs/complement-cosine-warmup-bs512.yaml
echo "===== Complement Done ====="

echo "===== All Done ====="
