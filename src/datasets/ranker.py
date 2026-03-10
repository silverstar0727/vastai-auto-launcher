import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

logger = logging.getLogger(__name__)

# Label constants (must match models/ranker.py)
NEUTRAL = -1
NEGATIVE = 0
POSITIVE = 1

TASK_LABEL_COLUMNS = {
    "ctr": "LABEL_CTR",
    "click_to_action": "LABEL_CLICK_TO_ACTION",
    "action_to_order": "LABEL_ACTION_TO_ORDER",
    "ctcar": "LABEL_CTCAR",
    "ctcvr": "LABEL_CTCVR",
    "cvr": "LABEL_CVR",
}


class RankerDataset(data_utils.Dataset):
    """Single-sample dataset for the multi-task ranker.

    Each sample returns:
        click_items: (seq_max_len,) padded item IDs from user click history
        item_features: (item_feature_dim,) feature vector for the candidate item
        labels: dict of task_name -> int label (NEUTRAL/NEGATIVE/POSITIVE)
        weights: dict of task_name -> float per-sample loss weight
    """

    def __init__(
        self,
        user_ids,
        item_ids,
        labels_df,
        click_history: dict,
        item_features: dict,
        item_feature_dim: int,
        seq_max_len: int = 200,
    ):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels_df = labels_df
        self.click_history = click_history
        self.item_features = item_features
        self.item_feature_dim = item_feature_dim
        self.seq_max_len = seq_max_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]

        # Click history: pad/truncate to seq_max_len
        history = self.click_history.get(user_id, [])
        if len(history) > self.seq_max_len:
            history = history[-self.seq_max_len :]
        pad_len = self.seq_max_len - len(history)
        click_items = [0] * pad_len + list(history)
        click_items = torch.LongTensor(click_items)

        # Item features
        if item_id in self.item_features:
            item_feat = torch.FloatTensor(self.item_features[item_id])
        else:
            item_feat = torch.zeros(self.item_feature_dim)

        # Labels
        row = self.labels_df.iloc[index]
        labels = {}
        weights = {}
        for task_name, col in TASK_LABEL_COLUMNS.items():
            if col in row.index:
                labels[task_name] = int(row[col])
            else:
                labels[task_name] = NEUTRAL

        # Per-sample weight (if present)
        weight_val = float(row["SAMPLE_WEIGHT"]) if "SAMPLE_WEIGHT" in row.index else 1.0
        for task_name in TASK_LABEL_COLUMNS:
            weights[task_name] = weight_val

        return click_items, item_feat, labels, weights


def _collate_fn(batch):
    """Custom collate to handle dict labels and weights."""
    click_items = torch.stack([b[0] for b in batch])
    item_features = torch.stack([b[1] for b in batch])

    task_names = list(batch[0][2].keys())
    labels = {
        task: torch.tensor([b[2][task] for b in batch], dtype=torch.long)
        for task in task_names
    }
    weights = {
        task: torch.tensor([b[3][task] for b in batch], dtype=torch.float32)
        for task in task_names
    }
    return click_items, item_features, labels, weights


class RankerDataModule(L.LightningDataModule):
    """DataModule for the multi-task ranker.

    Loads:
    - Label parquet files with USER_ID, ITEM_ID, task labels, is_train flag
    - Click history CSV files (user interaction sequences)
    - Item feature data (goods metadata with quantile-transformed stats)
    """

    def __init__(
        self,
        label_dir: str,
        interaction_dir: str,
        goods_path: str,
        item_embed_size: int = 256,
        market_embed_size: int = 128,
        category_embed_size: int = 64,
        seq_max_len: int = 200,
        batch_size: int = 10240,
        num_workers: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.label_dir = label_dir
        self.interaction_dir = interaction_dir
        self.goods_path = goods_path
        self.item_embed_size = item_embed_size
        self.market_embed_size = market_embed_size
        self.category_embed_size = category_embed_size
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self._is_setup = False

        # Populated during setup
        self.vocab_size = 0
        self.n_markets = 0
        self.n_categories = 0
        self.item_feature_dim = 0
        self.user_input_size = 0
        self.item_input_size = 0

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        logger.info("Loading labels...")
        labels_df = self._load_labels()
        logger.info(f"Loaded {len(labels_df)} label rows")

        logger.info("Loading click histories...")
        click_history = self._load_click_history()
        logger.info(f"Loaded click history for {len(click_history)} users")

        logger.info("Loading item features...")
        item_features, item_meta = self._load_item_features()
        logger.info(f"Loaded features for {len(item_features)} items")

        # Compute sizes from loaded data
        if item_features:
            sample_feat = next(iter(item_features.values()))
            self.item_feature_dim = len(sample_feat)
        self.item_input_size = self.item_feature_dim
        # user_input_size = seq_max_len (for embedded click sequence)
        # The actual user_input_size depends on how embeddings are aggregated
        # upstream; we expose it for external wiring.
        self.user_input_size = self.item_embed_size

        # Derive vocab/market/category sizes from metadata
        self.vocab_size = item_meta.get("vocab_size", 0)
        self.n_markets = item_meta.get("n_markets", 0)
        self.n_categories = item_meta.get("n_categories", 0)

        # Train/test split
        train_mask = labels_df["IS_TRAIN"] == 1
        train_df = labels_df[train_mask].reset_index(drop=True)
        test_df = labels_df[~train_mask].reset_index(drop=True)

        self.train_dataset = RankerDataset(
            user_ids=train_df["USER_ID"].tolist(),
            item_ids=train_df["ITEM_ID"].tolist(),
            labels_df=train_df,
            click_history=click_history,
            item_features=item_features,
            item_feature_dim=self.item_feature_dim,
            seq_max_len=self.seq_max_len,
        )
        self.test_dataset = RankerDataset(
            user_ids=test_df["USER_ID"].tolist(),
            item_ids=test_df["ITEM_ID"].tolist(),
            labels_df=test_df,
            click_history=click_history,
            item_features=item_features,
            item_feature_dim=self.item_feature_dim,
            seq_max_len=self.seq_max_len,
        )

        logger.info(f"Train samples: {len(self.train_dataset)}, Test samples: {len(self.test_dataset)}")
        self._is_setup = True

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    # --- Data Loading ---

    def _load_labels(self) -> pd.DataFrame:
        """Load label parquet files from label_dir."""
        parquet_files = sorted(glob(os.path.join(self.label_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.label_dir}")

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            dfs.append(df)
        labels_df = pd.concat(dfs, ignore_index=True)

        # Fill missing label columns with NEUTRAL
        for task_name, col in TASK_LABEL_COLUMNS.items():
            if col not in labels_df.columns:
                labels_df[col] = NEUTRAL

        # Ensure IS_TRAIN column exists
        if "IS_TRAIN" not in labels_df.columns:
            logger.warning("IS_TRAIN column not found; treating all data as train.")
            labels_df["IS_TRAIN"] = 1

        return labels_df

    def _load_click_history(self) -> dict:
        """Load user click history from CSV files.

        Returns:
            dict mapping USER_ID -> list of item IDs (chronologically ordered).
        """
        csv_files = sorted(glob(os.path.join(self.interaction_dir, "*.csv")))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.interaction_dir}")
            return {}

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
        interactions = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp and group by user
        if "TIMESTAMP" in interactions.columns:
            interactions = interactions.sort_values("TIMESTAMP")

        click_history = {}
        for user_id, group in interactions.groupby("USER_ID"):
            click_history[user_id] = group["ITEM_ID"].tolist()

        return click_history

    def _load_item_features(self) -> tuple:
        """Load item features from goods metadata.

        Returns:
            (item_features_dict, metadata_dict)
            item_features_dict: maps ITEM_ID -> numpy array of features
            metadata_dict: contains vocab_size, n_markets, n_categories
        """
        if not os.path.exists(self.goods_path):
            logger.warning(f"Goods file not found: {self.goods_path}")
            return {}, {"vocab_size": 0, "n_markets": 0, "n_categories": 0}

        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])

        vocab_size = len(df_goods)

        # Market and category counts
        n_markets = df_goods["market_sno"].nunique() if "market_sno" in df_goods.columns else 0
        n_categories = df_goods["category_sno"].nunique() if "category_sno" in df_goods.columns else 0

        # Extract numeric feature columns (quantile-transformed stats)
        numeric_cols = [
            c for c in df_goods.columns
            if c not in ("sno", "name", "market_sno", "category_sno", "market_name", "category_name")
            and df_goods[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        item_features = {}
        for _, row in df_goods.iterrows():
            item_id = row["sno"]
            feat = row[numeric_cols].values.astype(np.float32)
            item_features[item_id] = feat

        metadata = {
            "vocab_size": vocab_size,
            "n_markets": n_markets,
            "n_categories": n_categories,
        }

        return item_features, metadata
