"""SID Tokenizer DataModule.

입력:
  - meta/item_meta.parquet (goods_sno, name, price, category_sno, ...)
  - meta/standard_category.parquet
  - meta/active_items.parquet (goods_sno 풀)
  - embeddings/item_text_emb.npy + item_id_index.parquet
  - interactions/train/*.parquet (co-occurrence pair 추출)

Pair 정의: 같은 user 의 같은 dt 안에서 발생한 (item_a, item_b) — session 내 동시 등장.
PoC 효율을 위해 sampling 적용 (user 당 K 페어).

Validation: 전체 active items 한 번씩 통과 → SID uniqueness / recon 측정.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader


# -------- helpers --------

def _price_bucket(price: pd.Series, n_buckets: int) -> pd.Series:
    if n_buckets <= 0:
        return pd.Series(0, index=price.index)
    valid = price > 0
    qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
    q_vals = price[valid].quantile(qs).tolist()
    buckets = np.searchsorted(q_vals, price.fillna(0).values)
    buckets = np.clip(buckets, 0, n_buckets - 1) + 1  # 1..n_buckets, 0=pad
    buckets[price.isna() | (price <= 0)] = 0
    return pd.Series(buckets, index=price.index)


# -------- dataset --------

# vocab 컨벤션: 0 = PAD (시퀀스 빈 자리, gradient 안 흐름),
#               1 = UNK (정보 누락 — 의미 있는 학습 가능 토큰)
PAD_IDX = 0
UNK_IDX = 1


class _ItemFeatureStore:
    """item 한 개의 모든 feature 를 한 번에 조회 (torch tensor).

    vocab 처리:
      - category/brand 결측 → UNK_IDX (1). "정보 없음" 자체를 학습 가능 임베딩으로.
      - 미등록 sno (학습 vocab 밖) → UNK_IDX.
      - PAD (0) 은 시퀀스 padding 위치용으로 예약.
    """

    def __init__(
        self,
        item_meta_df: pd.DataFrame,
        text_emb: np.ndarray,
        text_emb_index: Dict[int, int],
        category_sno_to_idx: Dict[int, int],
        brand_sno_to_idx: Optional[Dict[int, int]] = None,
        num_price_buckets: int = 10,
        # === v3 신규 (백워드 호환: None → 기존 동작) ===
        attribute_lookup: Optional[Dict[int, np.ndarray]] = None,
        max_attributes_per_item: int = 16,
    ):
        self.text_emb = text_emb  # (N, D) fp16/32
        self.text_emb_index = text_emb_index  # goods_sno → row in text_emb

        self.category_sno_to_idx = category_sno_to_idx
        self.brand_sno_to_idx = brand_sno_to_idx or {}
        self.num_price_buckets = num_price_buckets
        self.attribute_lookup = attribute_lookup
        self.max_attributes_per_item = max_attributes_per_item

        item_meta_df = item_meta_df.copy()
        cat_series = pd.to_numeric(item_meta_df["standard_category_sno"], errors="coerce")
        brand_series = pd.to_numeric(item_meta_df["brand_sno"], errors="coerce")
        item_meta_df["goods_sno"] = item_meta_df["goods_sno"].astype("int64")

        self.goods_index: Dict[int, int] = {}
        cat_arr = []
        brand_arr = []
        for i, (sno, cat, brand) in enumerate(
            zip(item_meta_df["goods_sno"].values, cat_series.values, brand_series.values)
        ):
            self.goods_index[int(sno)] = i
            # cat: NaN/0/미등록 → UNK
            if pd.isna(cat) or cat <= 0:
                cat_arr.append(UNK_IDX)
            else:
                cat_arr.append(self.category_sno_to_idx.get(int(cat), UNK_IDX))
            # brand: NaN/0/미등록 → UNK
            if pd.isna(brand) or brand <= 0:
                brand_arr.append(UNK_IDX)
            else:
                brand_arr.append(self.brand_sno_to_idx.get(int(brand), UNK_IDX))
        self.category_id = np.array(cat_arr, dtype=np.int64)
        self.brand_id = np.array(brand_arr, dtype=np.int64)

        price_buckets = _price_bucket(item_meta_df["price"], num_price_buckets)
        self.price_bucket = price_buckets.values.astype(np.int64)

    def _attribute_tensor(self, goods_sno: int) -> torch.Tensor:
        """attribute_ids: (max_attributes_per_item,) — left/right padding 무관, PAD=0."""
        out = np.zeros(self.max_attributes_per_item, dtype=np.int64)
        if self.attribute_lookup is None:
            return torch.from_numpy(out)
        arr = self.attribute_lookup.get(int(goods_sno))
        if arr is not None and len(arr) > 0:
            k = min(len(arr), self.max_attributes_per_item)
            out[:k] = arr[:k]
        return torch.from_numpy(out)

    def get(self, goods_sno: int) -> Dict[str, torch.Tensor]:
        i = self.goods_index.get(int(goods_sno))
        if i is None:
            # 미등록 sno — text emb 도 없고 category/brand 도 UNK
            text = np.zeros(self.text_emb.shape[1], dtype=np.float32)
            return {
                "text_emb": torch.from_numpy(text),
                "category_id": torch.tensor(UNK_IDX, dtype=torch.long),
                "brand_id": torch.tensor(UNK_IDX, dtype=torch.long),
                "price_bucket": torch.tensor(UNK_IDX, dtype=torch.long),
                "attribute_ids": self._attribute_tensor(goods_sno),
            }
        text_row = self.text_emb_index.get(int(goods_sno))
        text = (
            self.text_emb[text_row].astype(np.float32)
            if text_row is not None
            else np.zeros(self.text_emb.shape[1], dtype=np.float32)
        )
        return {
            "text_emb": torch.from_numpy(text),
            "category_id": torch.tensor(self.category_id[i], dtype=torch.long),
            "brand_id": torch.tensor(self.brand_id[i], dtype=torch.long),
            "price_bucket": torch.tensor(self.price_bucket[i], dtype=torch.long),
            "attribute_ids": self._attribute_tensor(goods_sno),
        }


class _PairTrainDataset(Dataset):
    """(anchor, positive) pair 학습 데이터."""

    def __init__(self, pairs: np.ndarray, store: _ItemFeatureStore):
        self.pairs = pairs  # (M, 2)
        self.store = store

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, p = self.pairs[idx]
        return {
            "anchor": self.store.get(int(a)),
            "positive": self.store.get(int(p)),
        }


class _ItemValDataset(Dataset):
    """active items 한 번씩 통과 — SID uniqueness 측정."""

    def __init__(self, goods_snos: np.ndarray, store: _ItemFeatureStore):
        self.snos = goods_snos
        self.store = store

    def __len__(self):
        return len(self.snos)

    def __getitem__(self, idx):
        return self.store.get(int(self.snos[idx]))


# -------- datamodule --------

class SIDTokenizerDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        num_pairs_per_user: int = 4,
        max_train_pairs: int = 5_000_000,
        num_price_buckets: int = 10,
        batch_size: int = 4096,
        test_batch_size: int = 2048,
        num_workers: int = 4,
        seed: int = 42,
        # === v3 신규 (백워드 호환: False → 기존 동작) ===
        use_attributes: bool = False,
        max_attributes_per_item: int = 16,
        use_popularity_bias_pairs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.store: Optional[_ItemFeatureStore] = None
        self.num_categories: int = 0
        self.num_brands: int = 0
        self.num_attribute_values: int = 0  # 모델 setup() 이 읽음

    def prepare_data(self):
        # 외부 다운로드 등 없음
        return

    def setup(self, stage: Optional[str] = None):
        if self.store is not None:
            return
        meta = pd.read_parquet(self.base / "meta" / "item_meta.parquet")
        active = pd.read_parquet(self.base / "meta" / "active_items.parquet")
        active["goods_sno"] = pd.to_numeric(active["goods_sno"], errors="coerce").astype("int64")
        # active 만 사용
        meta["goods_sno"] = pd.to_numeric(meta["goods_sno"], errors="coerce").astype("int64")
        meta = meta[meta["goods_sno"].isin(active["goods_sno"])].reset_index(drop=True)

        # category vocab — index 0=PAD, 1=UNK, 2..N+1=실제 카테고리 sno
        cat_meta = pd.read_parquet(self.base / "meta" / "standard_category.parquet")
        cat_map = {int(s): i + 2 for i, s in enumerate(cat_meta["sno"].astype(int))}
        self.num_categories = len(cat_map) + 2  # PAD + UNK + 실제

        # brand vocab — active items 의 실제 brand_sno 만 (>0). 같은 규약.
        brand_series = pd.to_numeric(meta["brand_sno"], errors="coerce")
        brand_unique = sorted({int(b) for b in brand_series.dropna().tolist() if b > 0})
        brand_map = {b: i + 2 for i, b in enumerate(brand_unique)}
        self.num_brands = len(brand_map) + 2  # PAD + UNK + 실제

        # text emb 로드
        emb_path = self.base / "embeddings" / "item_text_emb.npy"
        index_path = self.base / "embeddings" / "item_id_index.parquet"
        if not emb_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "텍스트 임베딩이 아직 없음. Task #9 (09_extract_text_emb.py) 먼저 실행 필요."
            )
        text_emb = np.load(emb_path, mmap_mode="r")
        text_index_df = pd.read_parquet(index_path)
        text_index_df["goods_sno"] = pd.to_numeric(text_index_df["goods_sno"], errors="coerce").astype("int64")
        text_emb_index = dict(zip(text_index_df["goods_sno"], text_index_df["row_idx"]))

        # === v3: item_attributes 로드 (옵션) ===
        attribute_lookup = None
        if self.hparams.use_attributes:
            attr_path = self.base / "meta" / "item_attributes.parquet"
            ia = pd.read_parquet(attr_path, columns=["goods_sno", "value_sno"])
            ia["goods_sno"] = pd.to_numeric(ia["goods_sno"], errors="coerce").astype("int64")
            ia["value_sno"] = pd.to_numeric(ia["value_sno"], errors="coerce").astype("int64")
            ia = ia.dropna()
            # value_sno → contiguous index (1..N, 0=PAD)
            unique_values = sorted(ia["value_sno"].unique().tolist())
            value_to_idx = {int(v): i + 1 for i, v in enumerate(unique_values)}
            self.num_attribute_values = len(unique_values) + 1
            ia["value_idx"] = ia["value_sno"].map(value_to_idx).astype("int64")
            # goods_sno → list[value_idx]
            attribute_lookup = {}
            for sno, g in ia.groupby("goods_sno", sort=False):
                vals = g["value_idx"].to_numpy(dtype=np.int64)
                attribute_lookup[int(sno)] = vals
            print(
                f"[SID-DM] attributes loaded — values={self.num_attribute_values}, "
                f"items_covered={len(attribute_lookup):,}",
                flush=True,
            )

        self.store = _ItemFeatureStore(
            item_meta_df=meta,
            text_emb=text_emb,
            text_emb_index=text_emb_index,
            category_sno_to_idx=cat_map,
            brand_sno_to_idx=brand_map,
            num_price_buckets=self.hparams.num_price_buckets,
            attribute_lookup=attribute_lookup,
            max_attributes_per_item=self.hparams.max_attributes_per_item,
        )

        # 학습/검증 분할
        self.active_snos = meta["goods_sno"].values.astype(np.int64)

        # 학습 pair: train interactions 에서 추출
        self._train_pairs = self._build_train_pairs()

    def _build_train_pairs(self) -> np.ndarray:
        """train interactions 에서 (user, dt) 단위로 등장한 items 들끼리 sampling.

        v3: use_popularity_bias_pairs=True 면 pair 선택 시 item 빈도 역수 가중.
        """
        rng = np.random.default_rng(self.hparams.seed)
        active_set = set(self.active_snos.tolist())
        train_dir = self.base / "interactions" / "train"
        files = sorted(train_dir.glob("*.parquet"))
        pairs: List[np.ndarray] = []
        target = self.hparams.max_train_pairs

        # popularity bias: 전역 item 빈도 pre-pass
        item_freq: Dict[int, int] = {}
        if self.hparams.use_popularity_bias_pairs:
            print("[SID-DM] popularity bias pair sampling — item freq pre-pass...", flush=True)
            from collections import Counter
            counter: Counter = Counter()
            for fp in files:
                df = pd.read_parquet(fp, columns=["goods_sno"])
                counter.update(df["goods_sno"].astype("int64").to_numpy().tolist())
            item_freq = dict(counter)
            print(f"[SID-DM]   freq computed — {len(item_freq):,} unique items", flush=True)
        per_user = self.hparams.num_pairs_per_user
        for fp in files:
            if sum(len(p) for p in pairs) >= target:
                break
            df = pd.read_parquet(fp, columns=["user_code", "goods_sno"])
            df = df[df["goods_sno"].isin(active_set)]
            if df.empty:
                continue
            # user 별 grouping → 페어 sampling
            groups = df.groupby("user_code")["goods_sno"].apply(list)
            for items in groups:
                if len(items) < 2:
                    continue
                arr = np.array(items, dtype=np.int64)
                k = min(per_user, len(arr) * (len(arr) - 1) // 2)
                if self.hparams.use_popularity_bias_pairs and item_freq:
                    # weight ∝ 1/√freq, Word2Vec-style negative sampling correction
                    weights = np.array(
                        [1.0 / np.sqrt(max(item_freq.get(int(x), 1), 1)) for x in arr],
                        dtype=np.float64,
                    )
                    weights = weights / weights.sum()
                    idxs = rng.choice(len(arr), size=(k, 2), p=weights)
                else:
                    idxs = rng.integers(0, len(arr), size=(k, 2))
                idxs = idxs[idxs[:, 0] != idxs[:, 1]]
                if len(idxs) == 0:
                    continue
                pair = arr[idxs]
                pairs.append(pair)
        all_pairs = np.concatenate(pairs, axis=0)
        if len(all_pairs) > target:
            sel = rng.choice(len(all_pairs), target, replace=False)
            all_pairs = all_pairs[sel]
        return all_pairs

    def train_dataloader(self):
        ds = _PairTrainDataset(self._train_pairs, self.store)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        ds = _ItemValDataset(self.active_snos, self.store)
        return DataLoader(
            ds,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
