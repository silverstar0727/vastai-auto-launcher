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


# ---------------------------------------------------------------------------
# Stage 1: Two-Tower Dataset & DataModule
# ---------------------------------------------------------------------------


class TwoTowerDataset(data_utils.Dataset):
    """Dataset for two-tower pre-training.

    Each sample is a user's click history split 50/50:
        - First half: input click items (user features)
        - Second half: target items (supervision)
    """

    def __init__(
        self,
        click_sequences: np.ndarray,
        seq_max_len: int = 50,
        click_markets: np.ndarray = None,
        click_categories: np.ndarray = None,
        user_ages: np.ndarray = None,
    ):
        self.click_sequences = click_sequences
        self.seq_max_len = seq_max_len
        self.click_markets = click_markets
        self.click_categories = click_categories
        self.user_ages = user_ages

    def __len__(self):
        return len(self.click_sequences)

    def __getitem__(self, index):
        items = self.click_sequences[index]  # variable-length array of item IDs

        # Split 50/50: first half is input, second half is target
        mid = len(items) // 2
        input_items = items[:mid]
        target_items = items[mid:]

        # Pad/truncate input to seq_max_len
        input_padded = np.zeros(self.seq_max_len, dtype=np.int64)
        length = min(len(input_items), self.seq_max_len)
        input_padded[:length] = input_items[:length]

        user_features = {"click_items": torch.LongTensor(input_padded)}

        # Optional features for input items
        if self.click_markets is not None:
            markets = self.click_markets[index][:mid]
            market_padded = np.zeros(self.seq_max_len, dtype=np.int64)
            market_padded[:length] = markets[:length]
            user_features["click_markets"] = torch.LongTensor(market_padded)

        if self.click_categories is not None:
            cats = self.click_categories[index][:mid]
            cat_padded = np.zeros(self.seq_max_len, dtype=np.int64)
            cat_padded[:length] = cats[:length]
            user_features["click_categories"] = torch.LongTensor(cat_padded)

        if self.user_ages is not None:
            user_features["user_age"] = torch.FloatTensor([self.user_ages[index]])

        # Target: randomly sample one from the target half
        target_idx = np.random.randint(0, max(len(target_items), 1))
        target_item = target_items[target_idx] if len(target_items) > 0 else 0

        return user_features, torch.tensor(target_item, dtype=torch.long)


class TwoTowerDataModule(L.LightningDataModule):
    """DataModule for Two-Tower pre-training stage.

    Loads user interaction histories, filters users with >= min_clicks,
    and creates train/val splits.
    """

    def __init__(
        self,
        data_dir: str,
        seq_max_len: int = 50,
        min_clicks: int = 10,
        batch_size: int = 1024,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_max_len = seq_max_len
        self.min_clicks = min_clicks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._is_setup = False

    def setup(self, stage: str = None) -> None:
        if self._is_setup:
            return

        # Load interaction data
        csv_files = sorted(glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} interactions from {len(csv_files)} files")

        # Group by user, filter by min_clicks
        user_groups = df.groupby("user_id")
        click_sequences = []
        for user_id, group in user_groups:
            items = group["item_id"].values
            if len(items) >= self.min_clicks:
                click_sequences.append(items)

        logger.info(
            f"Filtered to {len(click_sequences)} users with >= {self.min_clicks} clicks"
        )

        # Train/val split
        train_seqs, val_seqs = train_test_split(
            click_sequences, test_size=0.1, random_state=self.seed
        )

        self.train_dataset = TwoTowerDataset(
            click_sequences=train_seqs,
            seq_max_len=self.seq_max_len,
        )
        self.val_dataset = TwoTowerDataset(
            click_sequences=val_seqs,
            seq_max_len=self.seq_max_len,
        )

        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        self._is_setup = True

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_two_tower,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_two_tower,
        )


def _collate_two_tower(batch):
    """Custom collate for TwoTowerDataset.

    Merges list of (user_features_dict, target_item) into batched tensors.
    """
    user_features_list, target_items = zip(*batch)

    # Stack user features
    user_features = {}
    for key in user_features_list[0]:
        user_features[key] = torch.stack([uf[key] for uf in user_features_list])

    target_items = torch.stack(target_items)
    return user_features, target_items


# ---------------------------------------------------------------------------
# Stage 2: Bandit Dataset & DataModule
# ---------------------------------------------------------------------------


class BanditDataset(data_utils.Dataset):
    """Dataset for bandit training from display logs.

    Each sample contains:
        - click_items: user's click history (for computing user embedding)
        - item_target:  displayed item index
        - label:        1 if clicked, 0 if not
    """

    def __init__(
        self,
        click_histories: np.ndarray,
        item_targets: np.ndarray,
        labels: np.ndarray,
        seq_max_len: int = 50,
    ):
        self.click_histories = click_histories
        self.item_targets = torch.LongTensor(item_targets)
        self.labels = torch.FloatTensor(labels)
        self.seq_max_len = seq_max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Pad/truncate click history
        items = self.click_histories[index]
        padded = np.zeros(self.seq_max_len, dtype=np.int64)
        length = min(len(items), self.seq_max_len)
        padded[:length] = items[:length]

        user_features = {"click_items": torch.LongTensor(padded)}
        return user_features, self.item_targets[index], self.labels[index]


class BanditDataModule(L.LightningDataModule):
    """DataModule for Neural Linear Bandit training (Stage 2).

    Loads display log data with click/no-click labels and user click histories.
    """

    def __init__(
        self,
        data_dir: str,
        seq_max_len: int = 50,
        min_clicks: int = 10,
        batch_size: int = 4096,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_max_len = seq_max_len
        self.min_clicks = min_clicks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._is_setup = False

    def setup(self, stage: str = None) -> None:
        if self._is_setup:
            return

        # Load display log data
        csv_files = sorted(glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} display log records from {len(csv_files)} files")

        # Need: user_id, item_id, label (click=1/no-click=0), plus user click history
        # Build click histories per user from interaction data
        interaction_dir = os.path.join(os.path.dirname(self.data_dir), "interaction")
        if os.path.isdir(interaction_dir):
            int_files = sorted(glob(os.path.join(interaction_dir, "*.csv")))
            int_dfs = [pd.read_csv(f) for f in int_files]
            int_df = pd.concat(int_dfs, ignore_index=True)
            user_click_map = int_df.groupby("user_id")["item_id"].apply(list).to_dict()
        else:
            # Fallback: build from display logs where label=1
            clicked = df[df["label"] == 1]
            user_click_map = clicked.groupby("user_id")["item_id"].apply(list).to_dict()

        # Filter users with enough clicks
        valid_users = {u for u, items in user_click_map.items() if len(items) >= self.min_clicks}
        df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} records from {len(valid_users)} users")

        # Build arrays
        click_histories = [
            np.array(user_click_map.get(uid, []), dtype=np.int64)
            for uid in df["user_id"].values
        ]
        item_targets = df["item_id"].values.astype(np.int64)
        labels = df["label"].values.astype(np.float32)

        # Train/val split
        train_idx, val_idx = train_test_split(
            np.arange(len(df)), test_size=0.1, random_state=self.seed, stratify=labels
        )

        self.train_dataset = BanditDataset(
            click_histories=[click_histories[i] for i in train_idx],
            item_targets=item_targets[train_idx],
            labels=labels[train_idx],
            seq_max_len=self.seq_max_len,
        )
        self.val_dataset = BanditDataset(
            click_histories=[click_histories[i] for i in val_idx],
            item_targets=item_targets[val_idx],
            labels=labels[val_idx],
            seq_max_len=self.seq_max_len,
        )

        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        self._is_setup = True

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_bandit,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_bandit,
        )


def _collate_bandit(batch):
    """Custom collate for BanditDataset.

    Merges list of (user_features_dict, item_target, label) into batched tensors.
    """
    user_features_list, item_targets, labels = zip(*batch)

    # Stack user features
    user_features = {}
    for key in user_features_list[0]:
        user_features[key] = torch.stack([uf[key] for uf in user_features_list])

    item_targets = torch.stack(item_targets)
    labels = torch.stack(labels)
    return user_features, item_targets, labels
