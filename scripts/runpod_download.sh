#!/bin/bash
# 모델별 5/6 일자(이전 날짜) 데이터를 S3 에서 받아온다.
# 사용법: runpod_download.sh <MODEL> <DATE> <DATA_ROOT>
set -euo pipefail

MODEL="${1:?MODEL required}"
DATE="${2:?DATE required (YYYYMMDD)}"
DATA="${3:-/data}"

log()  { echo "[download] $*"; }

mkdir -p "$DATA/sql" "$DATA/pretrained_text" "$DATA/bert/interaction"

# ---- 공통: pretrained_text (정적, 이미 있으면 sync 가 빠르게 끝남) ----
download_pretrained() {
    log "pretrained_text → $DATA/pretrained_text"
    aws s3 sync s3://ably-analytics-data/models/pretrained_text/ \
        "$DATA/pretrained_text/" --no-progress
}

# ---- 공통: 인터랙션 CSV (CBF / Complement / MI 가 같은 소스를 다른 로컬 경로로 사용) ----
download_interaction_to() {
    local target="$1"
    log "interaction CSV → $target"
    mkdir -p "$target"
    aws s3 sync "s3://ably-analytics-data/personalize/inhouse/$DATE/" \
        "$target/" --exclude "_SUCCESS" --no-progress
}

# ---- 공통: reco_raw_parquet 7종 ----
download_sql_parquets() {
    local with_attr="$1"   # "yes" | "no"
    local RECO=s3://ably-datalake/reco/db/reco_raw_parquet
    local tables="goods order category standard_category member"
    [ "$with_attr" = "yes" ] && tables="$tables goods_attribute_values goods_attribute_field_values"
    for tbl in $tables; do
        log "$tbl.parquet → $DATA/sql/"
        aws s3 cp "$RECO/$tbl/dt=$DATE/$tbl.parquet" "$DATA/sql/$tbl.parquet" --no-progress
    done
}

# ---- 모델별 분기 ----
case "$MODEL" in
    cbf-reprod|cbf-e[0-9]*|cbf3|cbf3-e[0-9]*)
        # cbf-reprod 및 운영 align 실험들 (cbf-e0 ~ cbf-e5 ...), cbf3 baseline + cbf3-e* 실험군
        # 전부 use_attr=True. 동일한 personalize/inhouse/<date>/ 인터랙션 + reco_raw_parquet 7종(+attr) 사용.
        download_pretrained
        download_interaction_to "$DATA/bert/interaction"
        download_sql_parquets "yes"
        ;;
    cbf|cbf-cosine-warmup-bs512)
        # 실험용 cbf.yaml 은 use_attr=False 라 attr parquet 불필요
        download_pretrained
        download_interaction_to "$DATA/bert/interaction"
        download_sql_parquets "no"
        ;;
    complement|complement-cosine-warmup-bs512)
        # complement 은 데이터 경로에 날짜 폴더 포함
        download_pretrained
        download_interaction_to "$DATA/complement/$DATE/interaction"
        download_sql_parquets "yes"
        ;;
    multi_interest)
        download_pretrained
        download_interaction_to "$DATA/bert/interaction"
        download_sql_parquets "no"
        ;;
    # ---------------------------------------------------------------
    # PoC v2_full production-matched 학습 (총 3종):
    #   - multi_interest_prod      Multi-Interest 단독 학습
    #   - tiger_lite_v7_prod       SID v3 + TIGER-lite v7 학습 (pipeline)
    #   - tiger_lite_letter_prod   HSTU CF teacher + LETTER SID + TIGER-lite v7 학습 (pipeline)
    #
    # 공통 다운로드:
    #   1) prod_1yr_train_data_v2/interaction/{date}/   (60일 aggregated CSV)
    #   2) prod_1yr_train_data_v2/order/{date}/
    #   3) reco_raw_parquet/{goods,order,category,...,goods_attribute_*}/dt={date}/
    #   4) 모델 artifacts (이전 단계 산출물 download — S3_MODEL_ARTIFACTS_PATH 지정 시):
    #      예: text 임베딩, item_to_sid, cf_embeddings
    # ---------------------------------------------------------------
    multi_interest_prod|tiger_lite_v7_prod|tiger_lite_letter_prod|sid_v2_v3_prod|sid_v2_letter_prod|hstu_two_tower_prod)
        log "[$MODEL] prod_1yr 60일 aggregated pool 다운로드"
        mkdir -p "$DATA/prod_1yr/interaction" "$DATA/prod_1yr/order"
        aws s3 sync "s3://ably-analytics-data/daily/prod_1yr_train_data_v2/interaction/$DATE/" \
            "$DATA/prod_1yr/interaction/" --exclude "_SUCCESS" --no-progress
        aws s3 sync "s3://ably-analytics-data/daily/prod_1yr_train_data_v2/order/$DATE/" \
            "$DATA/prod_1yr/order/" --exclude "_SUCCESS" --no-progress
        # SQL parquets (attribute 포함 — TIGER-lite SID v3 에서 item_attributes 사용)
        download_sql_parquets "yes"
        download_pretrained

        # 모델 artifacts 다운로드 (이전 단계 산출물 — 환경변수로 명시)
        # 예) tiger_lite_v7_prod 면 SID 학습 산출물 (meta_v3/item_to_sid.parquet) 필요
        #     tiger_lite_letter_prod 면 SID-LETTER 산출물 (meta_v3_letter/...) 필요
        #     sid_v2_letter_prod 면 HSTU CF embedding (cf_embeddings.pt) 필요
        if [ -n "${S3_MODEL_ARTIFACTS_PATH:-}" ]; then
            log "[$MODEL] 사전 산출물 다운로드: $S3_MODEL_ARTIFACTS_PATH → $DATA/v2_full_prod/"
            mkdir -p "$DATA/v2_full_prod"
            aws s3 sync "$S3_MODEL_ARTIFACTS_PATH" "$DATA/v2_full_prod/" --no-progress
        fi

        # 변환: prod CSV → v2_full schema parquet
        # /workspace 에서 git clone 한 코드 사용. python venv 는 bootstrap 에서 활성화됨.
        log "[$MODEL] prod CSV → v2_full schema 변환"
        python /workspace/scripts/prod_to_v2full_convert.py \
            --prod-root "$DATA/prod_1yr" \
            --sql-root "$DATA/sql" \
            --out-root "$DATA/v2_full_prod" \
            --date "$DATE"
        log "[$MODEL] prod_1yr → v2_full_prod 변환 완료"
        ;;
    lightgcn)
        # lightgcn 은 별도 train_events/valid_events parquet 사용 (사내 ETL 산출물)
        log "lightgcn 은 별도 ETL 산출물(train_events/valid_events) 필요."
        log "환경변수 LIGHTGCN_S3 가 지정돼있으면 그 경로에서 sync, 아니면 SKIP."
        if [ -n "${LIGHTGCN_S3:-}" ]; then
            mkdir -p "$DATA/lightgcn"
            aws s3 sync "$LIGHTGCN_S3" "$DATA/lightgcn/" --no-progress
        fi
        ;;
    *)
        echo "[download] WARN: $MODEL 에 대한 다운로드 정의 없음 — 그대로 진행" >&2
        ;;
esac

log "다운로드 완료"
