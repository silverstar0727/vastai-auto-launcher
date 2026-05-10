#!/bin/bash
# Runpod 컨테이너 entrypoint — runpod 전용
# 호스트에서 -v ~/.aws:/root/.aws:ro 로 SSO 캐시 마운트, -e WANDB_API_KEY 로 wandb 키 주입
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo "[entrypoint] $*"; }
fail() { echo "[entrypoint] ERROR: $*" >&2; exit 2; }

# ---- 1) 환경변수 검증 ----
MODEL="${MODEL:-cbf-reprod}"
DATE="${DATE:-}"
DATA_ROOT="${DATA_ROOT:-/data}"
CONFIG="configs/${MODEL}.yaml"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

[ -n "$DATE" ] || fail "DATE 환경변수가 비어있습니다 (예: -e DATE=20260506)"
[ -f "$CONFIG" ] || fail "config 파일 없음: $CONFIG"

log "MODEL=$MODEL DATE=$DATE DATA_ROOT=$DATA_ROOT"

# ---- 1.5) 이전 run 잔여 정리 (runpod container disk 가 stop/start 사이 영속이라 필요) ----
# /workspace/logs : lightning + wandb 로컬 dir. 남아있으면 _check_resume 가 죽은 run 의
#                   config.yaml 찾다 FileNotFoundError 로 학습 진입 자체를 차단함.
# $DATA_ROOT/output/$MODEL : preprocessed cache, encoder JSON 등.
# 환경변수 KEEP_LOGS=1 / KEEP_MODEL_PATH=1 로 각각 보존 가능.
if [ "${KEEP_LOGS:-0}" != "1" ] && [ -d /workspace/logs ]; then
    log "이전 /workspace/logs 정리 (KEEP_LOGS=1 로 비활성화)"
    rm -rf /workspace/logs
fi
if [ "${KEEP_MODEL_PATH:-0}" != "1" ] && [ -d "$DATA_ROOT/output/$MODEL" ]; then
    log "이전 $DATA_ROOT/output/$MODEL 정리 (KEEP_MODEL_PATH=1 로 비활성화)"
    rm -rf "$DATA_ROOT/output/$MODEL"
fi

mkdir -p "$DATA_ROOT" /workspace/logs

# ---- 2) AWS 인증 확인 ----
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "[2/5] AWS 인증 확인"
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        fail "AWS 인증 실패. AWS_ACCESS_KEY_ID 와 AWS_SECRET_ACCESS_KEY 환경변수가 설정됐는지 확인하세요."
    fi
    log "    OK ($(aws sts get-caller-identity --query Arn --output text 2>/dev/null))"
else
    log "[2/5] SKIP_DOWNLOAD=1 → 인증 검사 생략"
fi

# ---- 3) 데이터 다운로드 ----
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    log "[3/5] 데이터 다운로드 ($MODEL @ dt=$DATE)"
    bash "$SCRIPT_DIR/runpod_download.sh" "$MODEL" "$DATE" "$DATA_ROOT"
else
    log "[3/5] SKIP_DOWNLOAD=1 → 다운로드 생략 (DATA_ROOT 에 이미 데이터 있다고 가정)"
fi

# ---- 4) wandb 로그인 ----
log "[4/5] wandb 로그인"
if [ -n "${WANDB_API_KEY:-}" ]; then
    # wandb 바이너리 shebang 이슈 회피용으로 python -m 호출
    python -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 \
        && log "    wandb 로그인 OK" \
        || log "    wandb login 실패 (계속 진행, offline mode 가능)"
else
    log "    WANDB_API_KEY 비어있음. WANDB_MODE=offline 로 진행"
    export WANDB_MODE=offline
fi

# ---- 5) 학습 ----
# DATA_ROOT 가 yaml의 호스트 절대경로와 다를 때 자동으로 cmdline override 추가.
# 모델별로 키가 다르므로 cbf 계열만 처리. 다른 모델은 yaml 직접 수정 또는 "$@" 로 override.
EXTRA_OVERRIDES=()
case "$MODEL" in
    cbf|cbf-reprod|cbf-cosine-warmup-bs512|cbf-e[0-9]*)
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
    complement|complement-cosine-warmup-bs512)
        EXTRA_OVERRIDES=(
            "--data.init_args.raw_dataset_root=$DATA_ROOT/complement/$DATE/interaction"
            "--data.init_args.pretrained_root=$DATA_ROOT/pretrained_text"
            "--data.init_args.sql_dataset_root=$DATA_ROOT/sql"
            "--data.init_args.model_path=$DATA_ROOT/output/$MODEL"
            "--data.init_args.goods_filename=$DATA_ROOT/sql/goods.parquet"
            "--data.init_args.category_filename=$DATA_ROOT/sql/category.parquet"
            "--data.init_args.standard_category_filename=$DATA_ROOT/sql/standard_category.parquet"
            "--data.init_args.user_filename=$DATA_ROOT/sql/member.parquet"
            "--data.init_args.attr_meta_filename=$DATA_ROOT/sql/goods_attribute_field_values.parquet"
            "--data.init_args.attr_filename=$DATA_ROOT/sql/goods_attribute_values.parquet"
        )
        ;;
    multi_interest)
        EXTRA_OVERRIDES=(
            "--data.init_args.raw_dataset_root=$DATA_ROOT/bert/interaction"
            "--data.init_args.sql_dataset_root=$DATA_ROOT/sql"
            "--data.init_args.goods_filename=$DATA_ROOT/sql/goods.parquet"
            "--data.init_args.category_filename=$DATA_ROOT/sql/category.parquet"
            "--data.init_args.standard_category_filename=$DATA_ROOT/sql/standard_category.parquet"
            "--data.init_args.user_filename=$DATA_ROOT/sql/member.parquet"
            "--data.init_args.model_path=$DATA_ROOT/output/$MODEL"
        )
        ;;
esac

log "[5/5] 학습 시작: configs/${MODEL}.yaml"
exec python src/main.py fit -c "$CONFIG" "${EXTRA_OVERRIDES[@]}" "$@"
