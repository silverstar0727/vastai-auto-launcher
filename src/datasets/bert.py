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

# Map raw EVENT_TYPE strings to canonical event names
RAW_EVENT_MAP = {
    "click": "click",
    "view": "click",
    "like": "preference",
    "cart": "preference",
    "purchase": "preference",
    "deeplink": "preference",
    "order": "preference",
    "preference": "preference",
}

SECONDS_PER_WEEK = 7 * 24 * 3600


# --- Datasets ---


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, sequences, max_len, mask_prob, num_items, rng):
        self.sequences = sequences
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.mask_token = num_items + 1
        self.rng = rng

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])
        events = list(seq["events"])
        times = list(seq["times"])

        if len(items) > self.max_len:
            start = self.rng.randint(0, len(items) - self.max_len)
            items = items[start : start + self.max_len]
            events = events[start : start + self.max_len]
            times = times[start : start + self.max_len]

        tokens = []
        labels = []
        for i, (item_id, event) in enumerate(zip(items, events)):
            prob = self.rng.random()

            # preference items get 5x higher mask probability
            if event == EVENT_VOCA["preference"]:
                thd_prob = self.mask_prob * 5
            else:
                thd_prob = self.mask_prob

            if prob < thd_prob:
                prob /= thd_prob
                if prob < 0.8:
                    token = self.mask_token
                elif prob < 0.9:
                    token = self.rng.randint(1, self.num_items)
                else:
                    token = item_id
                tokens.append(token)
                labels.append(item_id)
                events[i] = EVENT_VOCA["click"]
            else:
                tokens.append(item_id)
                labels.append(0)

        pad_len = self.max_len - len(tokens)
        tokens = [0] * pad_len + tokens
        events = [0] * pad_len + events
        times = [0.0] * pad_len + times
        labels = [0] * pad_len + labels

        return (
            {
                "click_items": torch.LongTensor(tokens),
                "event_code": torch.LongTensor(events),
                "time_interval": torch.FloatTensor(times),
            },
            torch.LongTensor(labels),
        )


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, sequences, max_len, num_items):
        self.sequences = sequences
        self.max_len = max_len
        self.mask_token = num_items + 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])
        events = list(seq["events"])
        times = list(seq["times"])
        answer_items = seq["answer_items"]
        answer_events = seq["answer_events"]
        answer_times = seq["answer_times"]

        items = self._insert_pad_and_mask(items, self.mask_token, pad_token=0)
        events = self._insert_pad_and_mask(events, answer_events[0], pad_token=0)
        times = self._insert_pad_and_mask(times, answer_times[0], pad_token=0.0)

        return (
            {
                "click_items": torch.LongTensor(items),
                "event_code": torch.LongTensor(events),
                "time_interval": torch.FloatTensor(times),
            },
            torch.LongTensor(answer_items),
        )

    def _insert_pad_and_mask(self, seq, mask_token, pad_token=0):
        seq = seq + [mask_token]
        seq = seq[-self.max_len :]
        pad_len = self.max_len - len(seq)
        return [pad_token] * pad_len + seq


# --- DataModule ---


class BertDataModule(L.LightningDataModule):
    """BERT4Rec DataModule.

    Handles the full pipeline: loading interaction/order/goods data,
    filtering, label encoding, text embedding generation, train/test
    splitting, and dataset creation.
    """

    def __init__(
        self,
        interaction_dir: str,
        goods_path: str,
        category_path: str,
        pretrained_text_path: str,
        order_path: str = "",
        cache_dir: str = "",
        max_len: int = 80,
        mask_prob: float = 0.15,
        batch_size: int = 32,
        max_items: int = 180000,
        min_actions_per_user: int = 100,
        max_test_samples: int = 200000,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.goods_path = goods_path
        self.category_path = category_path
        self.pretrained_text_path = pretrained_text_path
        self.order_path = order_path
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.max_items = max_items
        self.min_actions_per_user = min_actions_per_user
        self.max_test_samples = max_test_samples
        self.num_workers = num_workers
        self.seed = seed

        self.num_items = 0
        self.text_embeddings = {}
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "bert_preprocessed.pkl").exists():
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

        # 8. Split by users into train/test sequences
        seq_len = self.max_len * 2
        train_seqs, test_seqs = self._split_by_users(df, seq_len)
        logger.info(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

        # 9. Generate text embeddings
        self.text_embeddings = self._generate_text_embeddings(on_voca_items)
        logger.info(f"Generated text embeddings for {len(self.text_embeddings)} items")

        # 10. Create datasets
        rng = random.Random(self.seed)
        self.train_dataset = BertTrainDataset(train_seqs, self.max_len, self.mask_prob, self.num_items, rng)
        self.val_dataset = BertEvalDataset(test_seqs, self.max_len, self.num_items)

    # --- Data Loading ---

    def _load_interactions(self) -> pd.DataFrame:
        dfs = []

        # Load interaction CSVs
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

    def _split_by_users(self, df, seq_len):
        rng = random.Random(self.seed)
        train_sequences = []
        test_sequences = []

        for _, group in df.groupby("USER_ID"):
            group = group.sort_values("TIMESTAMP")
            items = group["item_index"].values
            events = group["event_code"].values
            timestamps = group["TIMESTAMP"].values

            # Compute time intervals
            time_intervals = np.zeros(len(timestamps), dtype=np.float32)
            if len(timestamps) > 1:
                deltas = np.diff(timestamps.astype(np.float64))
                normalized = np.clip(deltas / SECONDS_PER_WEEK, 0.0, 1.0)
                time_intervals[1:] = normalized

            # Split into sub-sequences of seq_len
            n_splits = max(len(items) // (seq_len + 1), 1)
            for chunk_items, chunk_events, chunk_times in zip(
                np.array_split(items, n_splits),
                np.array_split(events, n_splits),
                np.array_split(time_intervals, n_splits),
            ):
                if len(chunk_items) < 2:
                    continue

                # Pick 1 random item as test (not the first item)
                if len(test_sequences) < self.max_test_samples and len(chunk_items) > 2:
                    test_idx = rng.randint(1, len(chunk_items) - 1)
                    test_sequences.append(
                        {
                            "items": np.delete(chunk_items, test_idx).tolist(),
                            "events": np.delete(chunk_events, test_idx).tolist(),
                            "times": np.delete(chunk_times, test_idx).tolist(),
                            "answer_items": [int(chunk_items[test_idx])],
                            "answer_events": [int(chunk_events[test_idx])],
                            "answer_times": [float(chunk_times[test_idx])],
                        }
                    )
                    train_items = np.delete(chunk_items, test_idx)
                    train_events = np.delete(chunk_events, test_idx)
                    train_times = np.delete(chunk_times, test_idx)
                else:
                    train_items = chunk_items
                    train_events = chunk_events
                    train_times = chunk_times

                train_sequences.append(
                    {
                        "items": train_items.tolist(),
                        "events": train_events.tolist(),
                        "times": train_times.tolist(),
                    }
                )

        return train_sequences, test_sequences

    # --- Text Embeddings ---

    def _generate_text_embeddings(self, on_voca_items) -> dict:
        from sentence_transformers import SentenceTransformer

        # Build item texts: "{category_name} {product_name}"
        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

        df_category = pd.read_csv(self.category_path)
        cat_map = df_category.set_index("sno")["catnm"].astype(str).to_dict()

        df_goods["category_name"] = df_goods["category_sno"].map(cat_map).fillna("")
        df_goods["item_text"] = df_goods["category_name"] + " " + df_goods["name"].astype(str)

        # Map item_id -> item_index
        df_goods["item_index"] = df_goods["sno"].astype(str).map(self.item_id2index)
        df_goods = df_goods.dropna(subset=["item_index"])
        df_goods["item_index"] = df_goods["item_index"].astype(int)

        # Encode texts
        logger.info(f"Encoding {len(df_goods)} item texts with SentenceTransformer...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = SentenceTransformer(self.pretrained_text_path)
        texts = df_goods["item_text"].tolist()
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
        del model
        torch.cuda.empty_cache()

        item_indices = df_goods["item_index"].tolist()
        return dict(zip(item_indices, embeddings))

    # --- Caching ---

    def _save_cache(self, cache_path: Path):
        data = {
            "num_items": self.num_items,
            "item_id2index": self.item_id2index,
            "text_embeddings": self.text_embeddings,
            "train_sequences": self.train_dataset.sequences,
            "test_sequences": self.val_dataset.sequences,
        }
        with open(cache_path / "bert_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached preprocessed data to {cache_path / 'bert_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "bert_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_items = data["num_items"]
        self.item_id2index = data["item_id2index"]
        self.text_embeddings = data["text_embeddings"]

        rng = random.Random(self.seed)
        self.train_dataset = BertTrainDataset(
            data["train_sequences"], self.max_len, self.mask_prob, self.num_items, rng
        )
        self.val_dataset = BertEvalDataset(data["test_sequences"], self.max_len, self.num_items)
