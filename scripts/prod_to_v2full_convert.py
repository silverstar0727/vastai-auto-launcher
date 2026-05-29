"""prod_1yr_train_data_v2 CSV → v2_full schema parquet 변환.

prod_1yr 의 60일 aggregated interaction CSV 와 order CSV, reco_raw_parquet 의
goods/category/standard_category/member 데이터를 합쳐서, 우리 v2_full PoC 와
동일한 schema (meta/*, interactions/{train,val,test}/*.parquet) 로 변환한다.

이렇게 변환된 데이터는 기존 SIDTokenizerDataModule / TIGERLiteDataModule /
MultiInterestProdDataModule 의 `base_dir` 만 swap 하면 그대로 사용 가능.

사용법:
    python scripts/prod_to_v2full_convert.py \
        --prod-root /data/prod_1yr \
        --sql-root /data/sql \
        --out-root /data/v2_full_prod \
        --date 20260506

산출 (out-root):
    meta/
        active_items.parquet      (goods_sno, popularity_count)
        item_meta.parquet         (goods_sno, name, price, category_sno, standard_category_sno, brand_sno, market_sno, ...)
        category.parquet          (catnm, depth, parent_category__sno, sno)
        standard_category.parquet (depth, parent_standard_category__sno, sno, std_category_name)
        item_attributes.parquet   (goods_sno, field_sno, value_sno, predict_confidence) — sql 에 있으면
        attribute_field_values.parquet (value_sno, value_name, app_display_name, field_sno)
        member_meta.parquet       (user_code, age, gender, ...)
    interactions/train/<DATE>.parquet   (전체 60일 → 단일 파일. user_code, goods_sno, event, ts)
    interactions/val/<DATE>.parquet     (빈 dir — production 은 leave-last-out 이라 별도 val 없음)
    interactions/test/<DATE>.parquet    (빈 dir — task #18 에서 별도 평가용)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# prod CSV schema (확인됨, 2026-02-15 기준):
#   USER_ID,ITEM_ID,TIMESTAMP,EVENT_TYPE
#   m10000788,34546068,1768880444,click
# EVENT_TYPE 은 string: click, like, cart, (purchase 는 order.csv 별도)
# 우리 v2_full schema 의 event 값과 동일 → mapping 불필요. 그대로 사용.
V2FULL_EVENTS = {"click", "like", "cart", "purchase"}


def convert_interactions(
    prod_interaction_dir: Path,
    out_root: Path,
    date: str,
    val_days: int = 3,
) -> tuple[int, int, int]:
    """prod interaction CSV → v2_full schema parquet 변환.

    단일 60일 aggregated 입력을 시간 기준으로 train (앞 ~57일) / val (last val_days) 로 분할.
    test 는 production-matched 학습에선 비워둠 (별도 task #18 에서 사용).

    Multi-Interest prod 학습은 DataModule 이 train+val 결합 후 leave-last-out → val 도 학습에 포함.
    TIGER-lite v7 학습은 train_days/val_days config 기준 분할 사용.
    """
    # prod 는 단일 거대 CSV (18.5GB). pyarrow 기반 chunk 처리.
    csv_files = sorted(prod_interaction_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"prod interaction CSV 없음: {prod_interaction_dir}")
    print(f"[convert] interaction CSV 파일: {len(csv_files)}", flush=True)

    # pyarrow streaming read 로 메모리 효율 ↑ (18.5GB 단일 file)
    import pyarrow.csv as pacsv

    chunks = []
    for f in csv_files:
        print(f"[convert] reading {f.name} ({f.stat().st_size/1e9:.2f}GB)", flush=True)
        # header 있음, 4 cols: USER_ID, ITEM_ID, TIMESTAMP, EVENT_TYPE
        table = pacsv.read_csv(
            str(f),
            read_options=pacsv.ReadOptions(use_threads=True),
            parse_options=pacsv.ParseOptions(delimiter=","),
            convert_options=pacsv.ConvertOptions(
                column_types={
                    "USER_ID": "string",
                    "ITEM_ID": "int64",
                    "TIMESTAMP": "int64",
                    "EVENT_TYPE": "string",
                },
            ),
        )
        chunks.append(table.to_pandas())
    df = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
    del chunks
    print(f"[convert] raw interactions: {len(df):,}", flush=True)

    # rename + filter
    df = df.rename(
        columns={"USER_ID": "user_code", "ITEM_ID": "goods_sno",
                 "TIMESTAMP": "ts", "EVENT_TYPE": "event"}
    )
    df = df.dropna(subset=["user_code", "goods_sno", "ts", "event"])
    df["goods_sno"] = df["goods_sno"].astype("int64")
    df["ts"] = df["ts"].astype("int64")
    # event 는 string 그대로 (v2_full 과 동일 vocab)
    df = df[df["event"].isin(V2FULL_EVENTS)]
    print(f"[convert] valid event 필터 후: {len(df):,}", flush=True)

    out = df[["user_code", "goods_sno", "event", "ts"]]

    # 시간 기준 train/val 분할 — ts quantile 사용
    val_ratio = val_days / 60.0   # prod 60일 가정
    ts_cut = out["ts"].quantile(1.0 - val_ratio)
    train_df = out[out["ts"] < ts_cut].copy()
    val_df = out[out["ts"] >= ts_cut].copy()

    out_train = out_root / "interactions" / "train" / f"{date}.parquet"
    out_val = out_root / "interactions" / "val" / f"{date}.parquet"
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    (out_root / "interactions" / "test").mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_train, index=False)
    val_df.to_parquet(out_val, index=False)

    n_users = out["user_code"].nunique()
    n_items = out["goods_sno"].nunique()
    print(
        f"[convert] train ({len(train_df):,}) + val ({len(val_df):,}, last ~{val_days}d) — "
        f"users={n_users:,}, items={n_items:,}",
        flush=True,
    )
    return len(out), n_users, n_items


def convert_meta(sql_root: Path, out_meta_dir: Path, active_items: set):
    """sql parquets → v2_full meta/ 디렉토리.

    필요한 파일:
      goods.parquet → item_meta.parquet (active item 만)
      category.parquet → category.parquet
      standard_category.parquet → standard_category.parquet
      member.parquet → member_meta.parquet
      goods_attribute_values.parquet → item_attributes.parquet (optional)
      goods_attribute_field_values.parquet → attribute_field_values.parquet (optional)
    """
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # active_items.parquet (interaction 에 등장한 goods_sno + popularity)
    active_df = pd.DataFrame({"goods_sno": sorted(active_items)})
    active_df.to_parquet(out_meta_dir / "active_items.parquet", index=False)
    print(f"[convert/meta] active_items: {len(active_df):,}", flush=True)

    # goods → item_meta
    goods = pd.read_parquet(sql_root / "goods.parquet")
    goods["goods_sno"] = pd.to_numeric(goods["goods_sno"], errors="coerce").astype("Int64")
    goods = goods.dropna(subset=["goods_sno"])
    goods["goods_sno"] = goods["goods_sno"].astype("int64")
    item_meta = goods[goods["goods_sno"].isin(active_items)].copy()
    item_meta.to_parquet(out_meta_dir / "item_meta.parquet", index=False)
    print(f"[convert/meta] item_meta: {len(item_meta):,}", flush=True)

    # category / standard_category / member 그대로 복사 (스키마 동일)
    for src, dst in [
        ("category.parquet", "category.parquet"),
        ("standard_category.parquet", "standard_category.parquet"),
        ("member.parquet", "member_meta.parquet"),
    ]:
        s = sql_root / src
        d = out_meta_dir / dst
        if s.exists():
            pd.read_parquet(s).to_parquet(d, index=False)
            print(f"[convert/meta] {src} → {dst}", flush=True)

    # attribute 데이터 (있으면)
    for src, dst in [
        ("goods_attribute_values.parquet", "item_attributes.parquet"),
        ("goods_attribute_field_values.parquet", "attribute_field_values.parquet"),
    ]:
        s = sql_root / src
        d = out_meta_dir / dst
        if s.exists():
            df = pd.read_parquet(s)
            # item_attributes 는 active item 만 keep
            if "goods_sno" in df.columns:
                df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce").astype("Int64")
                df = df.dropna(subset=["goods_sno"])
                df["goods_sno"] = df["goods_sno"].astype("int64")
                df = df[df["goods_sno"].isin(active_items)]
            df.to_parquet(d, index=False)
            print(f"[convert/meta] {src} → {dst}: {len(df):,}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prod-root", required=True, help="prod_1yr 다운 위치 (interaction/, order/)")
    p.add_argument("--sql-root", required=True, help="sql parquets 위치")
    p.add_argument("--out-root", required=True, help="v2_full_prod 출력 위치")
    p.add_argument("--date", required=True, help="YYYYMMDD")
    args = p.parse_args()

    prod_root = Path(args.prod_root)
    sql_root = Path(args.sql_root)
    out_root = Path(args.out_root)

    # 1. interactions/{train,val}/{date}.parquet 생성 (시간 분할)
    convert_interactions(prod_root / "interaction", out_root, args.date, val_days=3)

    # 2. active_items 추출 (train+val 에 등장한 goods_sno)
    train_p = pd.read_parquet(out_root / "interactions" / "train" / f"{args.date}.parquet", columns=["goods_sno"])
    val_p = pd.read_parquet(out_root / "interactions" / "val" / f"{args.date}.parquet", columns=["goods_sno"])
    active_items = set(train_p["goods_sno"].astype("int64").tolist()) | set(val_p["goods_sno"].astype("int64").tolist())
    print(f"[convert] active_items: {len(active_items):,}", flush=True)

    # 3. meta/
    convert_meta(sql_root, out_root / "meta", active_items)

    print(f"\n[convert] 완료 → {out_root}", flush=True)


if __name__ == "__main__":
    main()
