import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lightning as L

logger = logging.getLogger(__name__)

# --- Constants ---

PERF_FEATURES = [
    "cart_1y",
    "click_1y",
    "like_1y",
    "purchase_1y",
    "impression_1y",
    "cart_3m",
    "click_3m",
    "like_3m",
    "purchase_3m",
    "impression_3m",
    "price",
    "positive_review_count",
    "total_review_count",
    "pos_review_ratio",
]

# Additional pair-level features (set to 0 when not available)
PAIR_FEATURES = ["item_copurchase", "item_bt"]

ALL_FEATURES = PERF_FEATURES + PAIR_FEATURES


# --- Dataset ---


class ComplementaryV2Dataset(data_utils.Dataset):
    def __init__(self, features, source_ids, target_ids, labels):
        self.features = torch.FloatTensor(features)
        self.source_ids = torch.LongTensor(source_ids)
        self.target_ids = torch.LongTensor(target_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.source_ids[index],
            self.target_ids[index],
            self.labels[index],
        )


# --- DataModule ---


class ComplementaryV2DataModule(L.LightningDataModule):
    """DataModule for complementary_v2 (DCNv2 with item & category embeddings).

    Loads raw event parquets, joins with goods metadata for features,
    z-score normalizes, and creates stratified train/test splits.
    """

    def __init__(
        self,
        data_dir: str,
        goods_path: str,
        goods_df_path: str = "",
        zscore_path: str = "",
        batch_size: int = 20000,
        num_workers: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.goods_path = goods_path
        self.goods_df_path = goods_df_path
        self.zscore_path = zscore_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Populated during setup()
        self.num_items = 0
        self.num_categories = 0
        self.item_encoder = None
        self.category_encoder = None
        self.item_idx_to_category_idx = None
        self.feature_means = None
        self.feature_stds = None
        self._is_setup = False

    def setup(self, stage: str = None) -> None:
        if self._is_setup:
            return

        # 1. Load parquet data
        df = self._load_parquets()
        logger.info(f"Loaded {len(df)} samples from parquets")

        # 2. Ensure source/target are numeric (parquets may have string IDs)
        df["source"] = pd.to_numeric(df["source"], errors="coerce")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.dropna(subset=["source", "target"])
        df["source"] = df["source"].astype(int)
        df["target"] = df["target"].astype(int)

        # 3. Derive label from click/purchase columns if needed
        if "label" not in df.columns:
            df["label"] = ((df["click"] == 1) | (df["purchase"] == 1)).astype(float)
        else:
            df["label"] = df["label"].astype(float)

        # 4. Load goods metadata for features
        goods_df = self._load_goods_df()

        # 5. Join target features from goods metadata
        if goods_df is not None:
            goods_df = goods_df.set_index("goodsno")
            target_features = df["target"].map(
                lambda x: goods_df.loc[x, PERF_FEATURES].values
                if x in goods_df.index
                else np.zeros(len(PERF_FEATURES))
            )
            feature_matrix = np.stack(target_features.values).astype(np.float32)
        else:
            feature_matrix = np.zeros((len(df), len(PERF_FEATURES)), dtype=np.float32)

        # Add pair features (copurchase, bt) as zeros
        pair_zeros = np.zeros((len(df), len(PAIR_FEATURES)), dtype=np.float32)
        feature_matrix = np.concatenate([feature_matrix, pair_zeros], axis=1)

        # 6. Z-score normalization
        self._zscore_normalize(feature_matrix)

        # 7. Fit item encoder
        all_items = pd.concat([df["source"], df["target"]]).unique()
        self.item_encoder = LabelEncoder()
        self.item_encoder.fit(all_items)
        self.num_items = len(self.item_encoder.classes_)
        logger.info(f"Number of unique items: {self.num_items}")

        df["source_idx"] = self.item_encoder.transform(df["source"])
        df["target_idx"] = self.item_encoder.transform(df["target"])

        # 7. Build item-to-category mapping
        self._build_category_mapping(df)

        # 8. Stratified train/test split
        source_ids = df["source_idx"].values
        target_ids = df["target_idx"].values
        labels = df["label"].values.astype(np.float32)

        (
            feat_train, feat_test,
            src_train, src_test,
            tgt_train, tgt_test,
            lbl_train, lbl_test,
        ) = train_test_split(
            feature_matrix,
            source_ids,
            target_ids,
            labels,
            test_size=0.1,
            random_state=self.seed,
            stratify=labels,
        )

        logger.info(f"Train samples: {len(lbl_train)}, Test samples: {len(lbl_test)}")

        self.train_dataset = ComplementaryV2Dataset(feat_train, src_train, tgt_train, lbl_train)
        self.val_dataset = ComplementaryV2Dataset(feat_test, src_test, tgt_test, lbl_test)
        self._is_setup = True

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    # --- Internal helpers ---

    def _load_parquets(self) -> pd.DataFrame:
        parquet_files = sorted(glob(os.path.join(self.data_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        logger.info(f"Loading {len(parquet_files)} parquet files...")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    def _load_goods_df(self) -> pd.DataFrame | None:
        """Load goods metadata with per-item performance features."""
        if self.goods_df_path and os.path.exists(self.goods_df_path):
            logger.info(f"Loading goods_df from {self.goods_df_path}")
            return pd.read_pickle(self.goods_df_path)
        return None

    def _zscore_normalize(self, features: np.ndarray) -> None:
        """Z-score normalize features in-place."""
        import json

        if self.zscore_path and os.path.exists(self.zscore_path):
            with open(self.zscore_path) as f:
                zscore = json.load(f)
            means = np.array([zscore[k]["mean"] for k in ALL_FEATURES], dtype=np.float32)
            stds = np.array([zscore[k]["std"] for k in ALL_FEATURES], dtype=np.float32)
        else:
            means = features.mean(axis=0)
            stds = features.std(axis=0)

        stds[stds == 0] = 1.0
        self.feature_means = means
        self.feature_stds = stds
        features[:] = (features - means) / stds

    def _build_category_mapping(self, df: pd.DataFrame) -> None:
        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])

        # item_encoder.classes_ are ints; goods sno are also ints
        known_items = set(int(x) for x in self.item_encoder.classes_)
        df_goods = df_goods[df_goods["sno"].isin(known_items)]

        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(df_goods["category_sno"].values)
        self.num_categories = len(self.category_encoder.classes_)
        logger.info(f"Number of unique categories: {self.num_categories}")

        mapping = np.zeros(self.num_items, dtype=np.int64)
        for _, row in df_goods.iterrows():
            item_id = row["sno"]
            cat_id = row["category_sno"]
            try:
                item_idx = self.item_encoder.transform([item_id])[0]
                cat_idx = self.category_encoder.transform([cat_id])[0]
                mapping[item_idx] = cat_idx
            except ValueError:
                continue

        self.item_idx_to_category_idx = torch.LongTensor(mapping)
