#!/bin/bash
set -e

# ============================================================
# 로컬에서 Docker 빌드 후 Vast.ai 실행 스크립트
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 기본 설정
REGISTRY="${REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-lightning-training}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 사용법 출력
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

로컬에서 Docker 이미지를 빌드하고 Vast.ai에서 학습을 실행합니다.

Options:
    -r, --registry REGISTRY     Docker registry (default: docker.io)
    -n, --name NAME             Image name (default: lightning-training)
    -t, --tag TAG               Image tag (default: latest)
    -c, --config CONFIG         Config file path (default: configs/gnn_feature_engineered.yaml)
    -u, --username USERNAME     Docker Hub username (required for push)
    --gpu-type TYPE             GPU type (default: "RTX 4090")
    --max-price PRICE           Max price per hour (default: 2.0)
    --dry-run                   Build and search only, don't create instance
    --skip-build                Skip Docker build, use existing image
    --skip-push                 Skip Docker push (for local testing)
    -h, --help                  Show this help message

Environment Variables:
    VAST_API_KEY                Vast.ai API key (required)
    WANDB_API_KEY               Weights & Biases API key
    AWS_ACCESS_KEY_ID           AWS access key
    AWS_SECRET_ACCESS_KEY       AWS secret key
    AWS_DEFAULT_REGION          AWS region (default: ap-northeast-2)
    S3_DATA_PATH                S3 data path

Examples:
    # 기본 실행 (Docker Hub에 푸시)
    ./scripts/launch.sh -u myusername

    # Dry run (빌드만, 인스턴스 생성 안함)
    ./scripts/launch.sh -u myusername --dry-run

    # 특정 config로 실행
    ./scripts/launch.sh -u myusername -c configs/my_config.yaml

    # GPU 타입 변경
    ./scripts/launch.sh -u myusername --gpu-type "RTX 3090"
EOF
    exit 0
}

# 환경변수 확인
check_env() {
    local missing=()

    if [ -z "$VAST_API_KEY" ]; then
        missing+=("VAST_API_KEY")
    fi

    if [ -z "$WANDB_API_KEY" ]; then
        missing+=("WANDB_API_KEY")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Set them in your shell or create a .env file:"
        echo "  export VAST_API_KEY=your_vastai_api_key"
        echo "  export WANDB_API_KEY=your_wandb_api_key"
        exit 1
    fi
}

# 인자 파싱
DOCKER_USERNAME=""
GPU_TYPE="RTX 4090"
MAX_PRICE="2.0"
DRY_RUN=""
SKIP_BUILD=false
SKIP_PUSH=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -u|--username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --max-price)
            MAX_PRICE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Docker username 확인
if [ -z "$DOCKER_USERNAME" ] && [ "$SKIP_PUSH" = false ]; then
    log_error "Docker username is required for pushing. Use -u/--username or --skip-push"
    exit 1
fi

# 전체 이미지 경로 구성
if [ -n "$DOCKER_USERNAME" ]; then
    FULL_IMAGE="${REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# 메인 실행
main() {
    log_info "Starting local build and launch pipeline"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Image: $FULL_IMAGE"
    echo ""

    check_env

    cd "$PROJECT_DIR"

    # 1. Docker 빌드
    if [ "$SKIP_BUILD" = false ]; then
        log_info "Step 1/3: Building Docker image..."
        docker build -t "$FULL_IMAGE" .
        log_info "Build complete!"
    else
        log_warn "Step 1/3: Skipping Docker build"
    fi

    # 2. Docker 푸시
    if [ "$SKIP_PUSH" = false ]; then
        log_info "Step 2/3: Pushing Docker image to registry..."
        docker push "$FULL_IMAGE"
        log_info "Push complete!"
    else
        log_warn "Step 2/3: Skipping Docker push"
    fi

    # 3. Vast.ai 인스턴스 실행
    log_info "Step 3/3: Launching on Vast.ai..."
    echo ""

    python3 scripts/vastai_launcher.py \
        --docker-image "$FULL_IMAGE" \
        --config "$CONFIG_FILE" \
        --gpu-type "$GPU_TYPE" \
        --max-price "$MAX_PRICE" \
        --wandb-key "${WANDB_API_KEY:-}" \
        --aws-key "${AWS_ACCESS_KEY_ID:-}" \
        --aws-secret "${AWS_SECRET_ACCESS_KEY:-}" \
        --aws-region "${AWS_DEFAULT_REGION:-ap-northeast-2}" \
        --s3-path "${S3_DATA_PATH:-}" \
        $DRY_RUN \
        $EXTRA_ARGS

    echo ""
    log_info "Done!"
}

main
