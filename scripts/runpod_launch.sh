#!/bin/bash
# Runpod 에서 학습 pod 한 줄 띄우기.
#
# 흐름:
#   1) runpodctl 인증 확인
#   2) 사용자가 지정한 GPU/디스크/이미지/모델/날짜로 pod 생성
#   3) AWS 인증 정보는 runpod 콘솔의 Secret 에 등록한 IAM access key 를 사용
#      (template 의 환경변수 값에 `{{ RUNPOD_SECRET_aws_access_key_id }}` 처럼 참조)
#
# 사전 준비:
#   - runpodctl 설치: `wget -qO- cli.runpod.net | sudo bash`
#   - org 의 API key 등록: `runpodctl config --apiKey=<KEY>`     (org 별로 다른 key)
#   - runpod 콘솔 Secrets 메뉴에 다음 두 개 등록:
#       * aws_access_key_id      (IAM user access key)
#       * aws_secret_access_key  (IAM user secret)
#
# 사용법 예시:
#   WANDB_API_KEY=... bash scripts/runpod_launch.sh \
#     --gpu "NVIDIA RTX A6000" --gpu-count 1 --disk 200 --model cbf-reprod --date 20260506
#   bash scripts/runpod_launch.sh --list-gpus     (사용 가능한 GPU id 확인)
#   bash scripts/runpod_launch.sh --dry-run ...   (실제 호출 없이 명령만 출력)
set -euo pipefail

# ---------------- 기본값 (env / cmdline 으로 override) ----------------
DOCKERHUB_REPO="${DOCKERHUB_REPO:-silverstar456/ably-reco-lightning}"
TAG="${TAG:-latest}"

GPU_TYPE="${GPU_TYPE:-}"                      # 필수. `--list-gpus` 또는 `runpodctl pod gpu-types` 로 목록 확인
GPU_COUNT="${GPU_COUNT:-}"                    # 필수
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-}"    # 필수. 이미지 35GB + 데이터 ~25GB → 보통 100~150 권장
VOLUME_GB="${VOLUME_GB:-0}"                   # 선택. >0 일 때만 영속 볼륨 부착
VOLUME_MOUNT="${VOLUME_MOUNT:-/workspace}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"            # SECURE | COMMUNITY
DATA_CENTER_IDS="${DATA_CENTER_IDS:-}"        # 콤마구분 (예: "EU-RO-1,US-CA-2")
COUNTRY_CODE="${COUNTRY_CODE:-}"
TERMINATE_AFTER="${TERMINATE_AFTER:-24h}"     # 안전장치 — 학습 후 자동 종료

NAME="${NAME:-cbf-$(date +%y%m%d-%H%M)}"
MODEL="${MODEL:-}"                            # 필수
DATE="${DATE:-}"                              # 필수
WANDB_API_KEY_VAL="${WANDB_API_KEY:-}"
WANDB_PROJECT_VAL="${WANDB_PROJECT:-cbf-exp}"
# Runpod Secret 이름 (콘솔 Secrets 메뉴에서 등록한 키와 일치)
AWS_KEY_SECRET_NAME="${AWS_KEY_SECRET_NAME:-aws_access_key_id}"
AWS_SECRET_SECRET_NAME="${AWS_SECRET_SECRET_NAME:-aws_secret_access_key}"

DRY_RUN=0
LIST_GPUS=0

log() { echo "[launch] $*"; }
err() { echo "[launch] ERROR: $*" >&2; exit 2; }

# ---------------- cmdline 인자 파서 (선택) ----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU_TYPE="$2"; shift 2 ;;
        --gpu-count)   GPU_COUNT="$2"; shift 2 ;;
        --disk)        CONTAINER_DISK_GB="$2"; shift 2 ;;
        --volume)      VOLUME_GB="$2"; shift 2 ;;
        --image)       DOCKERHUB_REPO="${2%:*}"; TAG="${2##*:}"; [ "$2" = "$TAG" ] && TAG=latest; shift 2 ;;
        --name)        NAME="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --date)        DATE="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT_VAL="$2"; shift 2 ;;
        --cloud)       CLOUD_TYPE="$2"; shift 2 ;;
        --datacenter)  DATA_CENTER_IDS="$2"; shift 2 ;;
        --terminate-after) TERMINATE_AFTER="$2"; shift 2 ;;
        --list-gpus)   LIST_GPUS=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *) err "unknown arg: $1" ;;
    esac
done

# ---------------- runpodctl 사전 검증 ----------------
if ! command -v runpodctl >/dev/null; then
    if [ "$DRY_RUN" = "1" ]; then
        log "WARNING: runpodctl 미설치 — DRY_RUN 이라 진행"
    else
        err "runpodctl 설치 필요: wget -qO- cli.runpod.net | sudo bash"
    fi
fi

if [ "$LIST_GPUS" = "1" ]; then
    runpodctl get cloud --secureCloud=true 2>/dev/null || \
        runpodctl pod gpu-types 2>/dev/null || \
        runpodctl get gpu-types
    exit 0
fi

# ---------------- 필수 파라미터 검증 ----------------
MISSING=()
[ -n "$GPU_TYPE" ]          || MISSING+=("--gpu (or env GPU_TYPE)")
[ -n "$GPU_COUNT" ]         || MISSING+=("--gpu-count (or env GPU_COUNT)")
[ -n "$CONTAINER_DISK_GB" ] || MISSING+=("--disk (or env CONTAINER_DISK_GB)")
[ -n "$MODEL" ]             || MISSING+=("--model (or env MODEL)")
[ -n "$DATE" ]              || MISSING+=("--date (or env DATE)  YYYYMMDD")
[ -n "$WANDB_API_KEY_VAL" ] || MISSING+=("env WANDB_API_KEY")
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "[launch] ERROR: 필수 인자 누락:" >&2
    for m in "${MISSING[@]}"; do echo "    - $m" >&2; done
    echo "" >&2
    echo "  사용 예시:" >&2
    echo "    DATE=20260506 WANDB_API_KEY=... \\" >&2
    echo "      bash scripts/runpod_launch.sh \\" >&2
    echo "        --gpu \"NVIDIA RTX A6000\" --gpu-count 1 \\" >&2
    echo "        --disk 120 --model cbf-reprod" >&2
    exit 2
fi
# 숫자형 검증
[[ "$GPU_COUNT" =~ ^[0-9]+$ ]]         || err "GPU_COUNT 는 숫자여야 함 (받은 값: $GPU_COUNT)"
[[ "$CONTAINER_DISK_GB" =~ ^[0-9]+$ ]] || err "CONTAINER_DISK_GB 는 숫자여야 함 (받은 값: $CONTAINER_DISK_GB)"
[[ "$VOLUME_GB" =~ ^[0-9]+$ ]]         || err "VOLUME_GB 는 숫자여야 함 (받은 값: $VOLUME_GB)"
[[ "$DATE" =~ ^[0-9]{8}$ ]]            || err "DATE 는 YYYYMMDD 8자리 숫자여야 함 (받은 값: $DATE)"
# config 존재 (모델명 → yaml)
[ -f "$(dirname "$0")/../configs/${MODEL}.yaml" ] || err "configs/${MODEL}.yaml 없음. --model 값 확인."

# runpodctl API key 설정 여부 확인
if ! runpodctl config --help >/dev/null 2>&1 || \
   ! grep -q apiKey "$HOME/.runpod/config.toml" 2>/dev/null; then
    log "WARNING: runpodctl API key 미설정처럼 보임. 'runpodctl config --apiKey=<KEY>' 로 org 의 key 등록 후 다시 실행."
fi

# ---------------- AWS 인증은 runpod Secret 으로 처리 ----------------
# 호스트에서 키 추출 안 함. 컨테이너 안에서 runpod template 변수가 expand 되어
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY 환경변수로 들어옴.
log "AWS 인증: runpod Secret($AWS_KEY_SECRET_NAME, $AWS_SECRET_SECRET_NAME) 참조"

# ---------------- runpodctl 명령 조립 ----------------
CMD=(runpodctl create pod
    --name "$NAME"
    --image "${DOCKERHUB_REPO}:${TAG}"
    --gpu-id "$GPU_TYPE"
    --gpu-count "$GPU_COUNT"
    --cloud-type "$CLOUD_TYPE"
    --container-disk-in-gb "$CONTAINER_DISK_GB"
    --env "AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_${AWS_KEY_SECRET_NAME} }}"
    --env "AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_${AWS_SECRET_SECRET_NAME} }}"
    --env "AWS_DEFAULT_REGION=ap-northeast-2"
    --env "WANDB_API_KEY=$WANDB_API_KEY_VAL"
    --env "WANDB_PROJECT=$WANDB_PROJECT_VAL"
    --env "MODEL=$MODEL"
    --env "DATE=$DATE"
)

[ "$VOLUME_GB" -gt 0 ] && CMD+=(--volume-in-gb "$VOLUME_GB" --volume-mount-path "$VOLUME_MOUNT")
[ -n "$DATA_CENTER_IDS" ] && CMD+=(--data-center-ids "$DATA_CENTER_IDS")
[ -n "$COUNTRY_CODE" ] && CMD+=(--country-code "$COUNTRY_CODE")
[ -n "$TERMINATE_AFTER" ] && CMD+=(--terminate-after "$TERMINATE_AFTER")

# ---------------- 실행 ----------------
log "Pod 스펙:"
log "  name=$NAME"
log "  image=${DOCKERHUB_REPO}:${TAG}"
log "  gpu=$GPU_TYPE x $GPU_COUNT  cloud=$CLOUD_TYPE"
log "  disk=${CONTAINER_DISK_GB}GB${VOLUME_GB:+ + volume ${VOLUME_GB}GB}"
log "  model=$MODEL date=$DATE  wandb_project=$WANDB_PROJECT_VAL"
log "  terminate-after=$TERMINATE_AFTER"

if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN — 실제 호출 안 함. 아래 명령이 실행됐을 것 (WANDB_API_KEY 는 마스킹, AWS 는 runpod Secret 참조):"
    for tok in "${CMD[@]}"; do
        case "$tok" in
            WANDB_API_KEY=*) printf '  %q ' "WANDB_API_KEY=***" ;;
            *)               printf '  %q ' "$tok" ;;
        esac
    done
    echo
    exit 0
fi

"${CMD[@]}"
log "Pod 생성 요청 전송 완료. 'runpodctl get pod' 로 상태 확인."
