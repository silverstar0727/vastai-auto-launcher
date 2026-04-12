#!/bin/bash
set -e

cd /home/jeongmindo/projects/ably/reco-lightning

echo "===== Clean caches ====="
rm -rf /home/jeongmindo/projects/ably/data/output/cbf/preprocessed/
rm -rf /home/jeongmindo/projects/ably/data/output/complement/preprocessed/
rm -rf logs/cbf-cosine-warmup/ logs/complement-cosine-warmup/

echo "===== [1/2] CBF Training ====="
uv run python src/main.py fit -c configs/cbf.yaml
echo "===== CBF Done ====="

echo "===== [2/2] Complement Training ====="
uv run python src/main.py fit -c configs/complement.yaml
echo "===== Complement Done ====="

echo "===== All Done ====="
