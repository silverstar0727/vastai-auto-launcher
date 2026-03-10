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


class EmbeddingGeneratorTrainDataset(data_utils.Dataset):
    """Train dataset for embedding generator.

    For each user, 50% of history is sampled as input, the rest as positive labels.
    With negative_sampling: returns (inputs, pos_labels_tensor) with zero-padded labels.
    Without negative_sampling: returns (inputs, multi_hot_vector) over all items.
    """

    def __init__(self, sequences, max_len, num_items, rng, max_pos_items=400, negative_sampling=True):
        self.sequences = sequences
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.max_pos_items = max_pos_items
        self.negative_sampling = negative_sampling

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])

        n_input_samples = min(int(len(items) * 0.5), self.max_len)
        input_items = self.rng.sample(items, n_input_samples)
        output_items = np.setdiff1d(items, input_items)

        input_items = _insert_pad(input_items, self.max_len)

        input_tensors = {
            "click_items": torch.LongTensor(input_items),
        }

        # User side info
        if "click_markets" in seq:
            input_tensors["click_markets"] = torch.LongTensor(_insert_pad(seq["click_markets"], self.max_len))
        if "click_categories" in seq:
            input_tensors["click_categories"] = torch.LongTensor(_insert_pad(seq["click_categories"], self.max_len))
        if "user_age" in seq:
            input_tensors["user_age"] = torch.FloatTensor([seq["user_age"]])

        if self.negative_sampling:
            if len(output_items) > self.max_pos_items:
                pos_labels = self.rng.sample(output_items.tolist(), self.max_pos_items)
            else:
                pos_labels = _insert_pad(output_items.tolist(), self.max_pos_items)
            return input_tensors, torch.LongTensor(pos_labels)
        else:
            y = np.zeros((self.num_items + 1,))
            y[output_items] = 1.0
            return input_tensors, torch.FloatTensor(y)


class EmbeddingGeneratorBiasCorrectionTrainDataset(data_utils.Dataset):
    """Train dataset for embedding generator with bias correction.

    Returns (inputs, pos_labels, pos_probs) where pos_probs are the
    sampling probabilities for each positive item (for bias correction).
    """

    def __init__(self, sequences, max_len, num_items, rng, item_frequencies, max_pos_items=400):
        self.sequences = sequences
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.max_pos_items = max_pos_items
        self.sample_probs = self._compute_sample_probs(item_frequencies)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])

        n_input_samples = min(int(len(items) * 0.5), self.max_len)
        input_items = self.rng.sample(items, n_input_samples)
        output_items = np.setdiff1d(items, input_items)

        input_items = _insert_pad(input_items, self.max_len)

        input_tensors = {
            "click_items": torch.LongTensor(input_items),
        }
        if "click_markets" in seq:
            input_tensors["click_markets"] = torch.LongTensor(_insert_pad(seq["click_markets"], self.max_len))
        if "click_categories" in seq:
            input_tensors["click_categories"] = torch.LongTensor(_insert_pad(seq["click_categories"], self.max_len))
        if "user_age" in seq:
            input_tensors["user_age"] = torch.FloatTensor([seq["user_age"]])

        if len(output_items) > self.max_pos_items:
            pos_labels = self.rng.sample(output_items.tolist(), self.max_pos_items)
        else:
            pos_labels = _insert_pad(output_items.tolist(), self.max_pos_items)

        pos_probs = [max(self.sample_probs[x], 1e-7) for x in pos_labels]

        return input_tensors, torch.LongTensor(pos_labels), torch.FloatTensor(pos_probs)

    def _compute_sample_probs(self, item_frequencies):
        probs = np.zeros(self.num_items + 1)
        for item_idx, freq in item_frequencies.items():
            if item_idx < len(probs):
                probs[item_idx] = freq
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs


class EmbeddingGeneratorEvalDataset(data_utils.Dataset):
    """Eval dataset for embedding generator.

    Uses all history as input, test items as positive targets.
    """

    def __init__(self, sequences, max_len, max_answer_len=10):
        self.sequences = sequences
        self.max_len = max_len
        self.max_answer_len = max_answer_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = _insert_pad(list(seq["items"]), self.max_len)

        input_tensors = {
            "click_items": torch.LongTensor(items),
        }
        if "click_markets" in seq:
            input_tensors["click_markets"] = torch.LongTensor(_insert_pad(seq["click_markets"], self.max_len))
        if "click_categories" in seq:
            input_tensors["click_categories"] = torch.LongTensor(_insert_pad(seq["click_categories"], self.max_len))
        if "user_age" in seq:
            input_tensors["user_age"] = torch.FloatTensor([seq["user_age"]])

        answer_items = seq["answer_items"]
        if len(answer_items) < self.max_answer_len:
            answer_items = answer_items + [0] * (self.max_answer_len - len(answer_items))
        else:
            answer_items = answer_items[: self.max_answer_len]
        positive_items = torch.LongTensor(answer_items)
        return input_tensors, positive_items


# --- DataModule ---


class EmbeddingGeneratorDataModule(L.LightningDataModule):
    """Embedding Generator DataModule.

    Handles the full pipeline: loading interaction/goods data, filtering,
    label encoding, text embedding generation, train/test splitting,
    and dataset creation.
    """

    def __init__(
        self,
        interaction_dir: str,
        goods_path: str,
        category_path: str,
        standard_category_path: str,
        pretrained_text_path: str,
        order_path: str = "",
        cache_dir: str = "",
        max_len: int = 200,
        batch_size: int = 1024,
        test_batch_size: int = 1024,
        max_items: int = 1000000,
        min_actions_per_user: int = 50,
        max_test_samples: int = 200000,
        num_test_samples_per_user: int = 10,
        negative_sampling: bool = True,
        bias_correction: bool = False,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.goods_path = goods_path
        self.category_path = category_path
        self.standard_category_path = standard_category_path
        self.pretrained_text_path = pretrained_text_path
        self.order_path = order_path
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_items = max_items
        self.min_actions_per_user = min_actions_per_user
        self.max_test_samples = max_test_samples
        self.num_test_samples_per_user = num_test_samples_per_user
        self.negative_sampling = negative_sampling
        self.bias_correction = bias_correction
        self.num_workers = num_workers
        self.seed = seed

        self.num_items = 0
        self.text_embeddings = {}
        self.item_markets = None
        self.item_categories = None
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "embedding_generator_preprocessed.pkl").exists():
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

        # 7. Map IDs to indices
        df["item_index"] = df["ITEM_ID"].astype(str).map(self.item_id2index)
        df = df.dropna(subset=["item_index"])
        df["item_index"] = df["item_index"].astype(int)

        # 8. Build market/category label encoders
        n_markets, market_encoder = self._build_label_encoder(df, "MARKET_ID")
        n_categories, category_encoder = self._build_label_encoder(df, "STANDARD_CATEGORY_ID")
        self.n_markets = n_markets
        self.n_categories = n_categories

        # 9. Build item-level feature arrays
        self.item_markets, self.item_categories = self._build_item_features(
            df, market_encoder, category_encoder
        )

        # 10. Split by users into train/test sequences
        train_seqs, test_seqs, item_frequencies = self._split_by_users(
            df, market_encoder, category_encoder
        )
        logger.info(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

        # 11. Generate text embeddings
        self.text_embeddings = self._generate_text_embeddings(on_voca_items)
        logger.info(f"Generated text embeddings for {len(self.text_embeddings)} items")

        # 12. Create datasets
        rng = random.Random(self.seed)
        if self.bias_correction:
            self.train_dataset = EmbeddingGeneratorBiasCorrectionTrainDataset(
                train_seqs, self.max_len, self.num_items, rng, item_frequencies,
            )
        else:
            self.train_dataset = EmbeddingGeneratorTrainDataset(
                train_seqs, self.max_len, self.num_items, rng,
                negative_sampling=self.negative_sampling,
            )
        self.val_dataset = EmbeddingGeneratorEvalDataset(test_seqs, self.max_len, max_answer_len=self.num_test_samples_per_user)

    # --- Data Loading ---

    def _load_interactions(self) -> pd.DataFrame:
        dfs = []

        csv_files = sorted(glob(os.path.join(self.interaction_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.interaction_dir}")
        logger.info(f"Loading {len(csv_files)} interaction CSV files...")

        usecols = ["USER_ID", "ITEM_ID", "TIMESTAMP", "EVENT_TYPE"]
        # Try to load market/category columns if available
        optional_cols = ["MARKET_ID", "STANDARD_CATEGORY_ID", "USER_AGE"]

        sample_df = pd.read_csv(csv_files[0], nrows=0)
        available_cols = set(sample_df.columns)
        load_cols = usecols + [c for c in optional_cols if c in available_cols]

        for f in csv_files:
            df = pd.read_csv(f, usecols=load_cols)
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

    # --- Label Encoding ---

    def _build_label_encoder(self, df, column):
        """Build a 1-based label encoder for a column. Returns (n_labels, encoder_dict)."""
        if column not in df.columns:
            return 0, {}
        unique_vals = df[column].dropna().unique()
        encoder = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        return len(unique_vals), encoder

    def _build_item_features(self, df, market_encoder, category_encoder):
        """Build per-item market and category index arrays."""
        vocab_size = self.num_items + 1

        item_markets = np.zeros(vocab_size, dtype=np.int64)
        item_categories = np.zeros(vocab_size, dtype=np.int64)

        if market_encoder and "MARKET_ID" in df.columns:
            item_market_df = df.drop_duplicates(subset=["item_index"])[["item_index", "MARKET_ID"]]
            for _, row in item_market_df.iterrows():
                idx = int(row["item_index"])
                if idx < vocab_size:
                    item_markets[idx] = market_encoder.get(row["MARKET_ID"], 0)

        if category_encoder and "STANDARD_CATEGORY_ID" in df.columns:
            item_cat_df = df.drop_duplicates(subset=["item_index"])[["item_index", "STANDARD_CATEGORY_ID"]]
            for _, row in item_cat_df.iterrows():
                idx = int(row["item_index"])
                if idx < vocab_size:
                    item_categories[idx] = category_encoder.get(row["STANDARD_CATEGORY_ID"], 0)

        return item_markets, item_categories

    # --- Splitting ---

    def _split_by_users(self, df, market_encoder, category_encoder):
        rng = random.Random(self.seed)
        train_sequences = []
        test_sequences = []
        item_frequencies = {}

        has_markets = "MARKET_ID" in df.columns and bool(market_encoder)
        has_categories = "STANDARD_CATEGORY_ID" in df.columns and bool(category_encoder)
        has_age = "USER_AGE" in df.columns

        for _, group in df.groupby("USER_ID"):
            group = group.sort_values("TIMESTAMP")
            items = group["item_index"].values.tolist()

            # Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            items = unique_items

            if len(items) < 3:
                continue

            # Count item frequencies for bias correction
            for item in items:
                item_frequencies[item] = item_frequencies.get(item, 0) + 1

            # Pick test items (last N unique items)
            n_test = min(self.num_test_samples_per_user, len(items) // 2)
            if len(test_sequences) < self.max_test_samples and n_test > 0:
                test_items = items[-n_test:]
                train_items = items[:-n_test]
            else:
                test_items = []
                train_items = items

            # Build sequence dict
            seq_base = {}

            if has_markets:
                click_markets = []
                for item in train_items:
                    market_val = group[group["item_index"] == item]["MARKET_ID"].iloc[0] if len(group[group["item_index"] == item]) > 0 else None
                    click_markets.append(market_encoder.get(market_val, 0) if market_val is not None else 0)
                seq_base["click_markets"] = click_markets

            if has_categories:
                click_categories = []
                for item in train_items:
                    cat_val = group[group["item_index"] == item]["STANDARD_CATEGORY_ID"].iloc[0] if len(group[group["item_index"] == item]) > 0 else None
                    click_categories.append(category_encoder.get(cat_val, 0) if cat_val is not None else 0)
                seq_base["click_categories"] = click_categories

            if has_age:
                user_age = group["USER_AGE"].iloc[0]
                seq_base["user_age"] = float(user_age) if pd.notna(user_age) else 0.0

            train_seq = {"items": train_items}
            train_seq.update(seq_base)
            train_sequences.append(train_seq)

            if test_items:
                test_seq = {"items": train_items, "answer_items": test_items}
                test_seq.update(seq_base)
                test_sequences.append(test_seq)

        return train_sequences, test_sequences, item_frequencies

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
            "n_markets": self.n_markets,
            "n_categories": self.n_categories,
            "item_markets": self.item_markets,
            "item_categories": self.item_categories,
            "train_sequences": self.train_dataset.sequences,
            "test_sequences": self.val_dataset.sequences,
        }
        if hasattr(self.train_dataset, "sample_probs"):
            data["item_frequencies"] = {}  # reconstruct from sample_probs if needed
        with open(cache_path / "embedding_generator_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached preprocessed data to {cache_path / 'embedding_generator_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "embedding_generator_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_items = data["num_items"]
        self.item_id2index = data["item_id2index"]
        self.text_embeddings = data["text_embeddings"]
        self.n_markets = data.get("n_markets", 0)
        self.n_categories = data.get("n_categories", 0)
        self.item_markets = data.get("item_markets")
        self.item_categories = data.get("item_categories")

        rng = random.Random(self.seed)
        if self.bias_correction:
            item_frequencies = data.get("item_frequencies", {})
            self.train_dataset = EmbeddingGeneratorBiasCorrectionTrainDataset(
                data["train_sequences"], self.max_len, self.num_items, rng, item_frequencies,
            )
        else:
            self.train_dataset = EmbeddingGeneratorTrainDataset(
                data["train_sequences"], self.max_len, self.num_items, rng,
                negative_sampling=self.negative_sampling,
            )
        self.val_dataset = EmbeddingGeneratorEvalDataset(data["test_sequences"], self.max_len, max_answer_len=self.num_test_samples_per_user)


def _insert_pad(seq, max_len):
    """Left-pad sequence to max_len and truncate from the left if needed."""
    pad_len = max_len - len(seq)
    seq_padded = [0] * pad_len + seq
    seq_padded = seq_padded[-max_len:]
    return seq_padded
