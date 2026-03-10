import logging
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

import lightning as L

logger = logging.getLogger(__name__)

# --- Constants ---

FEATURES = [
    "src_cart_1y",
    "src_click_1y",
    "src_like_1y",
    "src_purchase_1y",
    "src_impression_1y",
    "src_cart_3m",
    "src_click_3m",
    "src_like_3m",
    "src_purchase_3m",
    "src_impression_3m",
    "tar_cart_1y",
    "tar_click_1y",
    "tar_like_1y",
    "tar_purchase_1y",
    "tar_impression_1y",
    "tar_cart_3m",
    "tar_click_3m",
    "tar_like_3m",
    "tar_purchase_3m",
    "tar_impression_3m",
]


# --- Dataset ---


class ComplementaryDataset(data_utils.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.weights[index]


# --- DataModule ---


class ComplementaryDataModule(L.LightningDataModule):
    """Complementary product prediction DataModule.

    Loads positive/negative parquet files, assigns labels,
    z-score normalizes features, and creates stratified train/test splits.
    """

    def __init__(
        self,
        pos_data_dir: str,
        neg_data_dir: str,
        zscore_path: str = "",
        batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.pos_data_dir = pos_data_dir
        self.neg_data_dir = neg_data_dir
        self.zscore_path = zscore_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._is_setup = False

    def setup(self, stage: str = None) -> None:
        if self._is_setup:
            return

        # 1. Load positive examples
        pos_files = sorted(glob(os.path.join(self.pos_data_dir, "*.parquet")))
        if not pos_files:
            raise FileNotFoundError(f"No parquet files found in {self.pos_data_dir}")
        df_pos = pd.concat([pd.read_parquet(f) for f in pos_files], ignore_index=True)
        df_pos["label"] = 1
        logger.info(f"Loaded {len(df_pos)} positive examples from {len(pos_files)} files")

        # 2. Load negative examples
        neg_files = sorted(glob(os.path.join(self.neg_data_dir, "*.parquet")))
        if not neg_files:
            raise FileNotFoundError(f"No parquet files found in {self.neg_data_dir}")
        df_neg = pd.concat([pd.read_parquet(f) for f in neg_files], ignore_index=True)
        df_neg["label"] = 0
        logger.info(f"Loaded {len(df_neg)} negative examples from {len(neg_files)} files")

        # 3. Combine
        df = pd.concat([df_pos, df_neg], ignore_index=True)

        # 4. Extract features and labels
        features = df[FEATURES].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)

        # 5. Z-score normalization
        if self.zscore_path and os.path.exists(self.zscore_path):
            zscore_df = pd.read_csv(self.zscore_path)
            means = zscore_df["mean"].values.astype(np.float32)
            stds = zscore_df["std"].values.astype(np.float32)
            logger.info(f"Using pre-computed z-score stats from {self.zscore_path}")
        else:
            means = features.mean(axis=0)
            stds = features.std(axis=0)
            logger.info("Computing z-score stats from data")

        stds = np.where(stds == 0, 1.0, stds)
        features = (features - means) / stds

        # 6. Compute sample weights (inverse class frequency)
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        n_total = len(labels)
        weight_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        weights = np.where(labels == 1, weight_pos, weight_neg).astype(np.float32)

        # 7. Stratified train/test split
        train_feat, test_feat, train_labels, test_labels, train_weights, test_weights = train_test_split(
            features,
            labels,
            weights,
            test_size=0.1,
            stratify=labels,
            random_state=self.seed,
        )
        logger.info(f"Train: {len(train_labels)}, Val: {len(test_labels)}")

        # 8. Create datasets
        self.train_dataset = ComplementaryDataset(train_feat, train_labels, train_weights)
        self.val_dataset = ComplementaryDataset(test_feat, test_labels, test_weights)
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
