import logging
import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

logger = logging.getLogger(__name__)

# --- Constants ---

EVENT_VOCA = {"unknown": 0, "click": 1, "preference": 2}

RAW_EVENT_MAP = {
    "click": "click",
    "view": "click",
    "like": "preference",
    "cart": "preference",
    "purchase": "preference",
    "deeplink": "click",
    "order": "preference",
    "preference": "preference",
}


# --- Datasets ---


class MultiInterestTrainDataset(data_utils.Dataset):
    """Training dataset for Multi-Interest model.

    For each user sequence, randomly picks a split point, uses items before
    as history (click_items, event_codes, standard_categories) and samples
    a positive item from the items after the split point.

    Preference events get higher sampling weight (preference_event_weight).
    """

    def __init__(
        self,
        sequences: list[dict],
        max_len: int,
        num_train_sample_per_user: int,
        preference_event_weight: float,
        item2category: np.ndarray,
        rng: random.Random,
    ):
        self.sequences = sequences
        self.max_len = max_len
        self.num_train_sample_per_user = num_train_sample_per_user
        self.preference_event_weight = preference_event_weight
        self.item2category = item2category
        self.rng = rng

    def __len__(self):
        return len(self.sequences) * self.num_train_sample_per_user

    def __getitem__(self, index):
        seq_idx = index // self.num_train_sample_per_user
        seq = self.sequences[seq_idx]

        items = list(seq["items"])
        events = list(seq["events"])
        n = len(items)

        # Pick a random split point: history = items[:split], future = items[split:]
        # At least 1 item in history and 1 item in future
        split = self.rng.randint(1, n - 1)

        history_items = items[:split]
        history_events = events[:split]

        # Sample a positive item from future window
        future_items = items[split:]
        future_events = events[split:]

        # Weighted sampling: preference events get higher weight
        weights = []
        for e in future_events:
            if e == EVENT_VOCA["preference"]:
                weights.append(self.preference_event_weight)
            else:
                weights.append(1.0)
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen_idx = self.rng.choices(range(len(future_items)), weights=probs, k=1)[0]
        positive_item = future_items[chosen_idx]

        # Truncate / pad history to max_len
        if len(history_items) > self.max_len:
            history_items = history_items[-self.max_len :]
            history_events = history_events[-self.max_len :]

        pad_len = self.max_len - len(history_items)
        padded_items = [0] * pad_len + history_items
        padded_events = [0] * pad_len + history_events

        # Look up standard category for each item
        padded_categories = [self.item2category[i] for i in padded_items]

        return (
            {
                "click_items": torch.LongTensor(padded_items),
                "event_codes": torch.LongTensor(padded_events),
                "standard_categories": torch.LongTensor(padded_categories),
            },
            torch.LongTensor([positive_item]),
        )


class MultiInterestEvalDataset(data_utils.Dataset):
    """Evaluation dataset for Multi-Interest model.

    Uses full history (minus held-out test item) to predict the test item.
    """

    def __init__(
        self,
        sequences: list[dict],
        max_len: int,
        item2category: np.ndarray,
    ):
        self.sequences = sequences
        self.max_len = max_len
        self.item2category = item2category

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])
        events = list(seq["events"])
        answer_items = seq["answer_items"]

        # Truncate history to max_len
        if len(items) > self.max_len:
            items = items[-self.max_len :]
            events = events[-self.max_len :]

        pad_len = self.max_len - len(items)
        padded_items = [0] * pad_len + items
        padded_events = [0] * pad_len + events
        padded_categories = [self.item2category[i] for i in padded_items]

        return (
            {
                "click_items": torch.LongTensor(padded_items),
                "event_codes": torch.LongTensor(padded_events),
                "standard_categories": torch.LongTensor(padded_categories),
            },
            torch.LongTensor(answer_items),
        )


# --- DataModule ---


class MultiInterestDataModule(L.LightningDataModule):
    """Multi-Interest DataModule.

    Handles the full pipeline: loading interaction/goods data, filtering,
    label encoding, standard category mapping, train/test splitting, and
    dataset creation.
    """

    def __init__(
        self,
        interaction_dir: str,
        goods_path: str,
        category_path: str,
        standard_category_path: str,
        order_path: str = "",
        cache_dir: str = "",
        max_len: int = 80,
        batch_size: int = 2048,
        test_batch_size: int = 64,
        max_items: int = 180000,
        min_actions_per_user: int = 100,
        max_test_samples: int = 200000,
        num_train_sample_per_user: int = 60,
        preference_event_weight: float = 5.0,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.goods_path = goods_path
        self.category_path = category_path
        self.standard_category_path = standard_category_path
        self.order_path = order_path
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_items = max_items
        self.min_actions_per_user = min_actions_per_user
        self.max_test_samples = max_test_samples
        self.num_train_sample_per_user = num_train_sample_per_user
        self.preference_event_weight = preference_event_weight
        self.num_workers = num_workers
        self.seed = seed

        self.num_items = 0
        self.num_categories = 0
        self.item2category: np.ndarray | None = None
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "multi_interest_preprocessed.pkl").exists():
            logger.info("Loading cached preprocessed data...")
            self._load_cache(cache_path)
        else:
            logger.info("Running full preprocessing pipeline...")
            self._run_preprocessing()
            if cache_path:
                cache_path.mkdir(parents=True, exist_ok=True)
                self._save_cache(cache_path)

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
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    # --- Preprocessing Pipeline ---

    def _run_preprocessing(self):
        # 1. Load raw interactions
        df = self._load_interactions()
        logger.info(f"Loaded {len(df)} raw interactions")

        # 2. Filter to on-sale items
        on_sale_items = self._get_on_sale_items()
        df = df[df["ITEM_ID"].isin(on_sale_items)]
        logger.info(f"After on-sale filter: {len(df)} interactions")

        # 3. Filter users with min actions
        df = self._filter_short_users(df, count_unique=False)
        logger.info(f"After min-user filter: {len(df)} interactions")

        # 4. Keep top max_items by popularity
        on_voca_items = self._get_top_items(df)
        df = df[df["ITEM_ID"].isin(on_voca_items)]
        logger.info(f"After vocab filter: {len(df)} interactions, {len(on_voca_items)} items")

        # 5. Re-filter users after vocab reduction
        df = self._filter_short_users(df, count_unique=True)
        logger.info(f"After re-filter: {len(df)} interactions")

        # 6. Create item encoder (item_id -> 1-based index)
        item_ids = df["ITEM_ID"].value_counts().index.tolist()
        self.item_id2index = {str(iid): idx + 1 for idx, iid in enumerate(item_ids)}
        self.item_id2index["unknown"] = 0
        self.num_items = len(item_ids)
        logger.info(f"Vocabulary size: {self.num_items}")

        # 7. Map IDs to indices and event codes
        df["item_index"] = df["ITEM_ID"].astype(str).map(self.item_id2index)
        df = df.dropna(subset=["item_index"])
        df["item_index"] = df["item_index"].astype(int)

        df["EVENT_TYPE"] = df["EVENT_TYPE"].map(RAW_EVENT_MAP).fillna("unknown")
        df["event_code"] = df["EVENT_TYPE"].map(EVENT_VOCA).fillna(0).astype(int)

        # 8. Build standard category mapping (item_index -> category_index)
        self.item2category, self.num_categories = self._build_category_map(on_voca_items)
        logger.info(f"Standard categories: {self.num_categories}")

        # 9. Split by users into train/test sequences
        train_seqs, test_seqs = self._split_by_users(df)
        logger.info(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

        # 10. Create datasets
        rng = random.Random(self.seed)
        self.train_dataset = MultiInterestTrainDataset(
            train_seqs,
            self.max_len,
            self.num_train_sample_per_user,
            self.preference_event_weight,
            self.item2category,
            rng,
        )
        self.val_dataset = MultiInterestEvalDataset(
            test_seqs,
            self.max_len,
            self.item2category,
        )

    # --- Data Loading ---

    def _load_interactions(self) -> pd.DataFrame:
        dfs = []

        csv_files = sorted(glob(os.path.join(self.interaction_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.interaction_dir}")
        logger.info(f"Loading {len(csv_files)} interaction CSV files...")
        for f in csv_files:
            df = pd.read_csv(f, usecols=["USER_ID", "ITEM_ID", "TIMESTAMP", "EVENT_TYPE"])
            dfs.append(df)

        # Load order data if provided
        if self.order_path and os.path.exists(self.order_path):
            logger.info("Loading order data...")
            df_order = pd.read_csv(self.order_path)
            df_order = df_order.rename(columns={"goods_sno": "ITEM_ID"})
            df_order["USER_ID"] = df_order["member_sno"].apply(lambda x: f"m{x}")
            df_order["TIMESTAMP"] = df_order["order_sno"] / 1000
            df_order["EVENT_TYPE"] = "order"
            dfs.append(df_order[["USER_ID", "ITEM_ID", "TIMESTAMP", "EVENT_TYPE"]])

        return pd.concat(dfs, ignore_index=True)

    def _get_on_sale_items(self) -> set:
        df_goods = pd.read_csv(self.goods_path, usecols=["sno"], escapechar="\\")
        return set(df_goods["sno"].tolist())

    # --- Category Mapping ---

    def _build_category_map(self, on_voca_items) -> tuple[np.ndarray, int]:
        """Build item_index -> standard_category_index mapping.

        Returns:
            item2category: ndarray of shape (num_items+1,) mapping item_index -> cat_index
            num_categories: number of unique standard categories (including padding 0)
        """
        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

        # Load standard category mapping
        df_std_cat = pd.read_csv(self.standard_category_path)

        # Map goods -> standard category
        # Try common column names; adjust if your schema differs
        if "standard_category_sno" in df_goods.columns:
            cat_col = "standard_category_sno"
        elif "category_sno" in df_goods.columns:
            # Need to map category_sno -> standard_category via the mapping table
            if "category_sno" in df_std_cat.columns and "standard_category_sno" in df_std_cat.columns:
                cat_mapping = df_std_cat.set_index("category_sno")["standard_category_sno"].to_dict()
                df_goods["standard_category_sno"] = df_goods["category_sno"].map(cat_mapping)
                cat_col = "standard_category_sno"
            else:
                # Fallback: use category_sno directly
                cat_col = "category_sno"
        else:
            # Use sno from standard_category_path directly
            cat_col = df_std_cat.columns[0]

        df_goods["item_index"] = df_goods["sno"].astype(str).map(self.item_id2index)
        df_goods = df_goods.dropna(subset=["item_index"])
        df_goods["item_index"] = df_goods["item_index"].astype(int)

        # Encode categories to 1-based index
        all_cats = df_goods[cat_col].dropna().unique().tolist()
        cat2index = {cat: idx + 1 for idx, cat in enumerate(all_cats)}
        num_categories = len(cat2index) + 1  # +1 for padding 0

        # Build lookup array
        item2category = np.zeros(self.num_items + 1, dtype=np.int64)
        for _, row in df_goods.iterrows():
            idx = int(row["item_index"])
            cat_val = row.get(cat_col)
            if pd.notna(cat_val) and cat_val in cat2index:
                item2category[idx] = cat2index[cat_val]

        return item2category, num_categories

    # --- Filtering ---

    def _filter_short_users(self, df, count_unique=False):
        if count_unique:
            counts = df.groupby("USER_ID")["ITEM_ID"].nunique()
        else:
            counts = df.groupby("USER_ID").size()
        good_users = counts[counts >= self.min_actions_per_user].index
        return df[df["USER_ID"].isin(good_users)]

    def _get_top_items(self, df) -> list:
        item_counts = df.groupby("ITEM_ID").size().reset_index(name="count")
        item_counts = item_counts.sort_values("count", ascending=False)
        if self.max_items and len(item_counts) > self.max_items:
            item_counts = item_counts.head(self.max_items)
        return item_counts["ITEM_ID"].tolist()

    # --- Splitting ---

    def _split_by_users(self, df):
        """Split each user's history into train and test sequences.

        For each user sub-sequence, hold out one random item as test target.
        """
        rng = random.Random(self.seed)
        train_sequences = []
        test_sequences = []

        for _, group in df.groupby("USER_ID"):
            group = group.sort_values("TIMESTAMP")
            items = group["item_index"].values.tolist()
            events = group["event_code"].values.tolist()

            if len(items) < 2:
                continue

            # Hold out one item for test (not the first item)
            if len(test_sequences) < self.max_test_samples and len(items) > 2:
                test_idx = rng.randint(1, len(items) - 1)
                test_item = items[test_idx]
                test_event = events[test_idx]

                train_items = items[:test_idx] + items[test_idx + 1 :]
                train_events = events[:test_idx] + events[test_idx + 1 :]

                test_sequences.append(
                    {
                        "items": train_items,
                        "events": train_events,
                        "answer_items": [test_item],
                    }
                )
            else:
                train_items = items
                train_events = events

            train_sequences.append(
                {
                    "items": train_items,
                    "events": train_events,
                }
            )

        return train_sequences, test_sequences

    # --- Caching ---

    def _save_cache(self, cache_path: Path):
        data = {
            "num_items": self.num_items,
            "num_categories": self.num_categories,
            "item_id2index": self.item_id2index,
            "item2category": self.item2category,
            "train_sequences": self.train_dataset.sequences,
            "test_sequences": self.val_dataset.sequences,
        }
        with open(cache_path / "multi_interest_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached preprocessed data to {cache_path / 'multi_interest_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "multi_interest_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_items = data["num_items"]
        self.num_categories = data["num_categories"]
        self.item_id2index = data["item_id2index"]
        self.item2category = data["item2category"]

        rng = random.Random(self.seed)
        self.train_dataset = MultiInterestTrainDataset(
            data["train_sequences"],
            self.max_len,
            self.num_train_sample_per_user,
            self.preference_event_weight,
            self.item2category,
            rng,
        )
        self.val_dataset = MultiInterestEvalDataset(
            data["test_sequences"],
            self.max_len,
            self.item2category,
        )
