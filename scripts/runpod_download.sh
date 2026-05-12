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
