#!/bin/bash
# Runpod 용 docker 이미지 빌드 + Docker Hub 푸시
# 사용법:
#   DOCKERHUB_USER=<username> bash scripts/runpod_build_push.sh
#   DOCKERHUB_USER=<username> IMAGE_NAME=ably-reco-lightning bash scripts/runpod_build_push.sh
#   DOCKERHUB_USER=<username> SKIP_PUSH=1 bash scripts/runpod_build_push.sh   # 빌드만
set -euo pipefail

# ---------- 설정 ----------
DOCKERHUB_USER="${DOCKERHUB_USER:?DOCKERHUB_USER 환경변수 필요 (예: DOCKERHUB_USER=myname)}"
IMAGE_NAME="${IMAGE_NAME:-ably-reco-lightning}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.runpod}"
SKIP_PUSH="${SKIP_PUSH:-0}"
PLATFORM="${PLATFORM:-linux/amd64}"   # runpod GPU = amd64

REPO="${DOCKERHUB_USER}/${IMAGE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

[ -f "$DOCKERFILE" ] || { echo "[build] $DOCKERFILE 없음" >&2; exit 2; }

# ---------- 태그 결정 ----------
GIT_SHA=""
if git rev-parse --short HEAD >/dev/null 2>&1; then
    GIT_SHA="$(git rev-parse --short HEAD)"
    if [ -n "$(git status --porcelain)" ]; then
        GIT_SHA="${GIT_SHA}-dirty"
    fi
fi
SHA_TAG="${GIT_SHA:-$(date -u +%Y%m%d-%H%M%S)}"

TAG_SHA="${REPO}:${SHA_TAG}"
TAG_LATEST="${REPO}:latest"

echo "[build] image      : $REPO"
echo "[build] tag (sha)  : $SHA_TAG"
echo "[build] tag (latest): latest"
echo "[build] dockerfile : $DOCKERFILE"
echo "[build] platform   : $PLATFORM"

# ---------- 빌드 ----------
DOCKER_BUILDKIT=1 docker build \
    --platform "$PLATFORM" \
    -f "$DOCKERFILE" \
    -t "$TAG_SHA" \
    -t "$TAG_LATEST" \
    "$PROJECT_DIR"

echo "[build] 빌드 완료"
docker images "$REPO" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}" | head -5

# ---------- 푸시 ----------
if [ "$SKIP_PUSH" = "1" ]; then
    echo "[build] SKIP_PUSH=1 → 푸시 건너뜀"
    exit 0
fi

# Docker Hub 로그인 확인
if ! docker info 2>/dev/null | grep -q "Username:"; then
    echo "[build] Docker Hub 로그인 필요. 'docker login' 먼저 수행."
    docker login docker.io
fi

echo "[build] $TAG_SHA 푸시 중..."
docker push "$TAG_SHA"
echo "[build] $TAG_LATEST 푸시 중..."
docker push "$TAG_LATEST"

echo "[build] 완료. 이미지 풀:"
echo "    docker pull $TAG_SHA"
echo "    docker pull $TAG_LATEST"
