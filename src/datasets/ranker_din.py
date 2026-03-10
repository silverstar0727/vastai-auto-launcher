import logging
import os
import random
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

logger = logging.getLogger(__name__)

# Neutral label code: sample is not applicable to this task
NEUTRAL = -1


def _pad_sequence(seq: list[int], max_len: int) -> list[int]:
    """Left-pad sequence with zeros and truncate to max_len."""
    pad_len = max_len - len(seq)
    padded = [0] * pad_len + seq
    return padded[-max_len:]


# --- Dataset ---


class RankerDinDataset(data_utils.Dataset):
    """Dataset for the ranker DIN model.

    Each sample yields:
        - user_features: (user_feat_dim,) non-click user feature vector
        - item_features: (item_feat_dim,) item feature vector
        - click_query: (1, click_embed_dim) candidate item embedding for DIN
        - click_keys: (seq_max_len, click_embed_dim) click history embeddings for DIN
        - click_mask: (seq_max_len,) boolean mask for valid click positions
        - labels: dict of per-task float labels (NEUTRAL=-1 for inapplicable)
        - weights: dict of per-task float loss weights
    """

    TASK_NAMES = ["ctr", "click_to_action", "ctcar", "ctcvr", "action_to_order", "cvr"]

    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        click_queries: np.ndarray,
        click_keys: np.ndarray,
        click_masks: np.ndarray,
        labels: dict[str, np.ndarray],
        weights: dict[str, np.ndarray],
        seq_max_len: int = 120,
        search_tokens: Optional[np.ndarray] = None,
        search_tokens_max_len: int = 80,
        rng: Optional[random.Random] = None,
        dropout_ratio: float = 0.1,
    ):
        """
        Args:
            user_features: (N, user_feat_dim) user feature matrix.
            item_features: (N, item_feat_dim) item feature matrix.
            click_queries: (N, click_embed_dim) candidate item embeddings.
            click_keys: (N, T, click_embed_dim) click history embeddings.
            click_masks: (N, T) boolean mask for valid click positions.
            labels: dict mapping task name to (N,) label arrays.
            weights: dict mapping task name to (N,) weight arrays.
            seq_max_len: Maximum click sequence length.
            search_tokens: Optional (N, search_tokens_max_len) search query token IDs.
            search_tokens_max_len: Maximum search token sequence length.
            rng: Random state for training-time augmentation. None for eval mode.
            dropout_ratio: Fraction of click history to drop during training.
        """
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
        self.click_queries = torch.FloatTensor(click_queries)
        self.click_keys = torch.FloatTensor(click_keys)
        self.click_masks = torch.BoolTensor(click_masks)
        self.labels = {k: torch.FloatTensor(v) for k, v in labels.items()}
        self.weights = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.seq_max_len = seq_max_len
        self.rng = rng
        self.dropout_ratio = dropout_ratio

        if search_tokens is not None:
            self.search_tokens = torch.LongTensor(search_tokens)
        else:
            self.search_tokens = None

    def __len__(self) -> int:
        return len(self.user_features)

    def __getitem__(self, index: int):
        user_feat = self.user_features[index]
        item_feat = self.item_features[index]
        click_q = self.click_queries[index].unsqueeze(0)  # (1, D)
        click_k = self.click_keys[index]  # (T, D)
        click_m = self.click_masks[index]  # (T,)

        labels = {name: self.labels[name][index] for name in self.TASK_NAMES}
        weights = {name: self.weights[name][index] for name in self.TASK_NAMES}

        return user_feat, item_feat, click_q, click_k, click_m, labels, weights


def _collate_fn(batch):
    """Custom collate that stacks tensors and merges label/weight dicts."""
    user_feats, item_feats, click_qs, click_ks, click_ms, labels_list, weights_list = zip(*batch)

    user_feats = torch.stack(user_feats)
    item_feats = torch.stack(item_feats)
    click_qs = torch.stack(click_qs)
    click_ks = torch.stack(click_ks)
    click_ms = torch.stack(click_ms)

    task_names = labels_list[0].keys()
    labels = {name: torch.stack([l[name] for l in labels_list]) for name in task_names}
    weights = {name: torch.stack([w[name] for w in weights_list]) for name in task_names}

    return user_feats, item_feats, click_qs, click_ks, click_ms, labels, weights


# --- DataModule ---


class RankerDinDataModule(L.LightningDataModule):
    """DataModule for the ranker DIN model.

    Loads preprocessed numpy/parquet data, splits into train/val sets,
    and creates DataLoaders with the custom collate function.
    """

    def __init__(
        self,
        data_dir: str,
        seq_max_len: int = 120,
        search_tokens_max_len: int = 80,
        batch_size: int = 30000,
        num_workers: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_max_len = seq_max_len
        self.search_tokens_max_len = search_tokens_max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._is_setup = False

    def setup(self, stage: str = None) -> None:
        if self._is_setup:
            return

        logger.info(f"Loading ranker DIN data from {self.data_dir}")

        # Load pre-computed arrays
        user_features = np.load(os.path.join(self.data_dir, "user_features.npy"))
        item_features = np.load(os.path.join(self.data_dir, "item_features.npy"))
        click_queries = np.load(os.path.join(self.data_dir, "click_queries.npy"))
        click_keys = np.load(os.path.join(self.data_dir, "click_keys.npy"))
        click_masks = np.load(os.path.join(self.data_dir, "click_masks.npy"))

        # Load labels
        labels = {}
        weights = {}
        for name in RankerDinDataset.TASK_NAMES:
            labels[name] = np.load(os.path.join(self.data_dir, f"label_{name}.npy"))
            weight_path = os.path.join(self.data_dir, f"weight_{name}.npy")
            if os.path.exists(weight_path):
                weights[name] = np.load(weight_path)
            else:
                weights[name] = np.ones_like(labels[name], dtype=np.float32)

        # Optional search tokens
        search_tokens_path = os.path.join(self.data_dir, "search_tokens.npy")
        search_tokens = np.load(search_tokens_path) if os.path.exists(search_tokens_path) else None

        # Load train/val split indices
        train_idx_path = os.path.join(self.data_dir, "train_indices.npy")
        val_idx_path = os.path.join(self.data_dir, "val_indices.npy")
        if os.path.exists(train_idx_path) and os.path.exists(val_idx_path):
            train_idx = np.load(train_idx_path)
            val_idx = np.load(val_idx_path)
        else:
            # Default 90/10 split
            n = len(user_features)
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(n)
            split = int(0.9 * n)
            train_idx = perm[:split]
            val_idx = perm[split:]

        def _slice(arr, idx):
            return arr[idx]

        def _slice_dict(d, idx):
            return {k: v[idx] for k, v in d.items()}

        rng = random.Random(self.seed)
        self.train_dataset = RankerDinDataset(
            user_features=_slice(user_features, train_idx),
            item_features=_slice(item_features, train_idx),
            click_queries=_slice(click_queries, train_idx),
            click_keys=_slice(click_keys, train_idx),
            click_masks=_slice(click_masks, train_idx),
            labels=_slice_dict(labels, train_idx),
            weights=_slice_dict(weights, train_idx),
            seq_max_len=self.seq_max_len,
            search_tokens=_slice(search_tokens, train_idx) if search_tokens is not None else None,
            search_tokens_max_len=self.search_tokens_max_len,
            rng=rng,
        )
        self.val_dataset = RankerDinDataset(
            user_features=_slice(user_features, val_idx),
            item_features=_slice(item_features, val_idx),
            click_queries=_slice(click_queries, val_idx),
            click_keys=_slice(click_keys, val_idx),
            click_masks=_slice(click_masks, val_idx),
            labels=_slice_dict(labels, val_idx),
            weights=_slice_dict(weights, val_idx),
            seq_max_len=self.seq_max_len,
            search_tokens=_slice(search_tokens, val_idx) if search_tokens is not None else None,
            search_tokens_max_len=self.search_tokens_max_len,
            rng=None,
        )

        logger.info(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )
