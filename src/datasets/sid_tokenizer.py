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

class _ItemFeatureStore:
    """item 한 개의 모든 feature 를 한 번에 조회 (torch tensor)."""

    def __init__(
        self,
        item_meta_df: pd.DataFrame,
        text_emb: np.ndarray,
        text_emb_index: Dict[int, int],
        category_sno_to_idx: Dict[int, int],
        brand_sno_to_idx: Optional[Dict[int, int]] = None,
        num_price_buckets: int = 10,
    ):
        self.text_emb = text_emb  # (N, D) fp16/32
        self.text_emb_index = text_emb_index  # goods_sno → row in text_emb

        self.category_sno_to_idx = category_sno_to_idx
        self.brand_sno_to_idx = brand_sno_to_idx or {}
        self.num_price_buckets = num_price_buckets

        # vectorized lookup
        self.goods_index: Dict[int, int] = {}
        cat_arr = []
        brand_arr = []
        for i, row in enumerate(item_meta_df.itertuples(index=False)):
            self.goods_index[int(row.goods_sno)] = i
            cat_arr.append(self.category_sno_to_idx.get(int(row.standard_category_sno or 0), 0))
            brand_arr.append(self.brand_sno_to_idx.get(int(row.brand_sno or 0), 0))
        self.category_id = np.array(cat_arr, dtype=np.int64)
        self.brand_id = np.array(brand_arr, dtype=np.int64)

        price_buckets = _price_bucket(item_meta_df["price"], num_price_buckets)
        self.price_bucket = price_buckets.values.astype(np.int64)

    def get(self, goods_sno: int) -> Dict[str, torch.Tensor]:
        i = self.goods_index.get(int(goods_sno))
        if i is None:
            # padding-like dummy (zero text emb + 0 category/brand/price)
            text = np.zeros(self.text_emb.shape[1], dtype=np.float32)
            return {
                "text_emb": torch.from_numpy(text),
                "category_id": torch.tensor(0, dtype=torch.long),
                "brand_id": torch.tensor(0, dtype=torch.long),
                "price_bucket": torch.tensor(0, dtype=torch.long),
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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.store: Optional[_ItemFeatureStore] = None
        self.num_categories: int = 0
        self.num_brands: int = 0

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

        # category vocab
        cat_meta = pd.read_parquet(self.base / "meta" / "standard_category.parquet")
        cat_map = {int(s): i + 1 for i, s in enumerate(cat_meta["sno"].astype(int))}
        self.num_categories = len(cat_map) + 1

        # brand vocab (현재 active items 에 등장하는 brand_sno 한정)
        brand_unique = sorted(set(meta["brand_sno"].fillna(0).astype(int).tolist()))
        brand_map = {b: i + 1 for i, b in enumerate(brand_unique) if b > 0}
        self.num_brands = len(brand_map) + 1

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

        self.store = _ItemFeatureStore(
            item_meta_df=meta,
            text_emb=text_emb,
            text_emb_index=text_emb_index,
            category_sno_to_idx=cat_map,
            brand_sno_to_idx=brand_map,
            num_price_buckets=self.hparams.num_price_buckets,
        )

        # 학습/검증 분할
        self.active_snos = meta["goods_sno"].values.astype(np.int64)

        # 학습 pair: train interactions 에서 추출
        self._train_pairs = self._build_train_pairs()

    def _build_train_pairs(self) -> np.ndarray:
        """train interactions 에서 (user, dt) 단위로 등장한 items 들끼리 sampling."""
        rng = np.random.default_rng(self.hparams.seed)
        active_set = set(self.active_snos.tolist())
        train_dir = self.base / "interactions" / "train"
        files = sorted(train_dir.glob("*.parquet"))
        pairs: List[np.ndarray] = []
        target = self.hparams.max_train_pairs
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
                # per_user 만큼 random pair
                k = min(per_user, len(arr) * (len(arr) - 1) // 2)
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
