#!/bin/bash
# 로컬 호스트에서 도커 없이 cbf / cbf3 / 기타 모델을 처음부터 끝까지 돌리는 스크립트.
# runpod_entrypoint.sh 와 같은 단계를 거치지만:
#   - bind-mount 대신 호스트 경로 직접 사용 (yaml 의 hardcoded /home/jeongmindo/... 그대로)
#   - AWS SSO 는 호스트의 AWS_PROFILE 사용 (마운트 없음)
#   - wandb 는 호스트의 ~/.netrc 사용
#   - uv 가상환경 그대로 사용 (uv sync 안 함)
#   - pod self-terminate 없음 (호스트라 종료할 pod 없음)
#
# 사용법:
#   MODEL=cbf3 ./scripts/local_run.sh                          # 첫 실행 (download + train)
#   MODEL=cbf3-e1 SKIP_DOWNLOAD=1 ./scripts/local_run.sh       # cache 사용
#   GPU_ID=1 MODEL=cbf3-e4 SKIP_DOWNLOAD=1 ./scripts/local_run.sh   # GPU 1 핀
#
# 환경변수:
#   MODEL              (필수) configs/${MODEL}.yaml
#   DATE               (default: 20260506)
#   DATA_ROOT          (default: $HOME/projects/ably/data — yaml hardcoded base)
#   AWS_PROFILE        (default: MLEngineer-145481888492)
#   SKIP_DOWNLOAD      (0/1, default 0)
#   KEEP_LOGS          (0/1, default 0 — 1 이면 logs/$MODEL/ 보존)
#   KEEP_MODEL_PATH    (0/1, default 0 — 1 이면 $DATA_ROOT/output/$MODEL/ 보존)
#   GPU_ID             (선택, 정수. 지정 시 CUDA_VISIBLE_DEVICES=$GPU_ID)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log()  { echo "[local_run] $*"; }
fail() { echo "[local_run] ERROR: $*" >&2; exit 2; }

# ---- 1) 환경변수 검증 ----
: "${MODEL:?MODEL required (예: cbf3, cbf3-e1, cbf-e4, ...)}"
export DATE="${DATE:-20260506}"
export DATA_ROOT="${DATA_ROOT:-$HOME/projects/ably/data}"
export AWS_PROFILE="${AWS_PROFILE:-MLEngineer-145481888492}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-northeast-2}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

CONFIG="$PROJECT_DIR/configs/${MODEL}.yaml"
[ -f "$CONFIG" ] || fail "config 없음: $CONFIG"

log "MODEL=$MODEL DATE=$DATE DATA_ROOT=$DATA_ROOT"
log "AWS_PROFILE=$AWS_PROFILE"

cd "$PROJECT_DIR"

# ---- 1.1) GPU 핀 ----
if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    log "GPU: physical GPU $GPU_ID 만 사용 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

# ---- 1.5) 이전 run 잔여 정리 ----
if [ "${KEEP_LOGS:-0}" != "1" ] && [ -d "$PROJECT_DIR/logs/$MODEL" ]; then
    log "이전 logs/$MODEL 정리 (KEEP_LOGS=1 로 비활성화)"
    rm -rf "$PROJECT_DIR/logs/$MODEL"
fi
if [ "${KEEP_MODEL_PATH:-0}" != "1" ] && [ -d "$DATA_ROOT/output/$MODEL" ]; then
    log "이전 $DATA_ROOT/output/$MODEL 정리 (KEEP_MODEL_PATH=1 로 비활성화)"
    rm -rf "$DATA_ROOT/output/$MODEL"
fi

mkdir -p "$DATA_ROOT" "$PROJECT_DIR/logs"

# ---- 2) AWS SSO 인증 확인 ----
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "[2/4] AWS SSO 인증 확인 (profile: $AWS_PROFILE)"
    if ! aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null 2>&1; then
        log "    SSO 만료/미인증. 자동 로그인 시도 (브라우저 열림)"
        aws sso login --profile "$AWS_PROFILE"
        aws sts get-caller-identity --profile "$AWS_PROFILE" >/dev/null \
            || fail "SSO 재로그인 후에도 인증 실패"
    fi
    log "    OK ($(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Arn --output text 2>/dev/null))"
else
    log "[2/4] SKIP_DOWNLOAD=1 → 인증 검사 생략"
fi

# ---- 3) 데이터 다운로드 ----
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "[3/4] 데이터 다운로드 ($MODEL @ dt=$DATE)"
    # runpod_download.sh 가 aws CLI 만 쓰면 호스트에서도 그대로 동작.
    bash "$SCRIPT_DIR/runpod_download.sh" "$MODEL" "$DATE" "$DATA_ROOT"
else
    log "[3/4] SKIP_DOWNLOAD=1 → 다운로드 생략"
fi

# ---- 4) 학습 ----
log "[4/4] 학습 시작: $CONFIG"
# 호스트는 yaml 의 hardcoded 경로 (/home/jeongmindo/projects/ably/data/*) 가 그대로 맞으므로
# DATA_ROOT override 불필요. 단 사용자가 DATA_ROOT 를 다른 곳으로 지정한 경우에만 override.
EXTRA_OVERRIDES=()
DEFAULT_DATA_ROOT="$HOME/projects/ably/data"
if [ "$DATA_ROOT" != "$DEFAULT_DATA_ROOT" ]; then
    log "DATA_ROOT=$DATA_ROOT (yaml default 와 다름) → cmdline override 추가"
    case "$MODEL" in
        cbf|cbf-reprod|cbf-cosine-warmup-bs512|cbf-e[0-9]*|cbf3|cbf3-e[0-9]*)
            EXTRA_OVERRIDES=(
                "--data.init_args.raw_dataset_root=$DATA_ROOT/bert/interaction"
                "--data.init_args.pretrained_root=$DATA_ROOT/pretrained_text"
                "--data.init_args.sql_dataset_root=$DATA_ROOT/sql"
                "--data.init_args.model_path=$DATA_ROOT/output/$MODEL"
                "--data.init_args.goods_filename=$DATA_ROOT/sql/goods.parquet"
                "--data.init_args.category_filename=$DATA_ROOT/sql/category.parquet"
                "--data.init_args.standard_category_filename=$DATA_ROOT/sql/standard_category.parquet"
                "--data.init_args.user_filename=$DATA_ROOT/sql/member.parquet"
                "--data.init_args.order_filename=$DATA_ROOT/sql/order.parquet"
                "--data.init_args.attr_meta_filename=$DATA_ROOT/sql/goods_attribute_field_values.parquet"
                "--data.init_args.attr_filename=$DATA_ROOT/sql/goods_attribute_values.parquet"
            )
            ;;
    esac
fi

# uv 가상환경의 python 사용. uv sync 는 이미 끝났다고 가정.
exec uv run python src/main.py fit -c "$CONFIG" "${EXTRA_OVERRIDES[@]}" "$@"
