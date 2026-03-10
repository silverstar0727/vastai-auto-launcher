import logging
import os
import pickle
import random
from glob import glob
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

logger = logging.getLogger(__name__)

# --- Constants ---

# Raw events preserved for deny condition filtering (before final mapping)
RAW_EVENT_MAP = {
    "click": "click",
    "like": "like",
    "cart": "cart",
    "order": "order",
    "view": "click",
    "purchase": "order",
    "query": "query",
}

# Final event mapping: order/cart/like -> preference, click -> click
FINAL_EVENT_MAP = {
    "click": "click",
    "like": "preference",
    "cart": "preference",
    "order": "preference",
}


# --- Datasets ---


class ComplementTrainDataset(data_utils.Dataset):
    """Complement training dataset.

    For each user, randomly sample 50% of history as input and the other 50% as targets.
    With negative_sampling=True, returns positive item indices (zero-padded).
    Otherwise, returns multi-hot label vector over all items.
    """

    def __init__(self, sequences, max_len, num_items, rng, max_pos_items=400, negative_sampling=False):
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

        # Split 50% input, 50% target
        n_input = min(int(len(items) * 0.5), self.max_len)
        input_items = self.rng.sample(items, n_input)
        output_items = list(np.setdiff1d(items, input_items))

        # Pad input to max_len (left padding with 0)
        input_items = _insert_pad(input_items, self.max_len)

        input_tensors = {
            "click_items": torch.LongTensor(input_items),
        }

        if self.negative_sampling:
            if len(output_items) > self.max_pos_items:
                pos_labels = self.rng.sample(output_items, self.max_pos_items)
            else:
                pos_labels = _insert_pad(output_items, self.max_pos_items)
            return input_tensors, torch.LongTensor(pos_labels)
        else:
            y = np.zeros(self.num_items + 1, dtype=np.float32)
            y[output_items] = 1.0
            return input_tensors, torch.FloatTensor(y)


class ComplementEvalDataset(data_utils.Dataset):
    """Complement evaluation dataset.

    Uses all train history as input, test items as targets.
    """

    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        items = list(seq["items"])
        answer_items = seq["answer_items"]

        items = _insert_pad(items, self.max_len)

        input_tensors = {
            "click_items": torch.LongTensor(items),
        }

        return input_tensors, torch.LongTensor(answer_items)


# --- DataModule ---


class ComplementDataModule(L.LightningDataModule):
    """Complement model DataModule.

    Extended data pipeline compared to CBF:
    - Loads interactions + order data
    - Applies deny condition filter (clickbait: clicks>=30K AND events/clicks<0.02)
    - Dual vocabulary: interaction items (top 180K) + order/cart items (top 400K)
    - Event mapping: order/cart/like -> preference
    - 50% input / 50% target split per user
    """

    def __init__(
        self,
        interaction_dir: str,
        order_path: str,
        goods_path: str,
        category_path: str,
        standard_category_path: str,
        pretrained_text_path: str,
        cache_dir: str = "",
        max_len: int = 100,
        batch_size: int = 256,
        max_interaction_items: int = 180000,
        max_ordered_items: int = 400000,
        min_actions_per_user: int = 50,
        max_test_samples: int = 200000,
        num_test_samples_per_user: int = 10,
        deny_condition_min_click: int = 30000,
        deny_condition_events_per_click: float = 0.02,
        negative_sampling: bool = False,
        max_pos_items: int = 400,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.order_path = order_path
        self.goods_path = goods_path
        self.category_path = category_path
        self.standard_category_path = standard_category_path
        self.pretrained_text_path = pretrained_text_path
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_interaction_items = max_interaction_items
        self.max_ordered_items = max_ordered_items
        self.min_actions_per_user = min_actions_per_user
        self.max_test_samples = max_test_samples
        self.num_test_samples_per_user = num_test_samples_per_user
        self.deny_condition_min_click = deny_condition_min_click
        self.deny_condition_events_per_click = deny_condition_events_per_click
        self.negative_sampling = negative_sampling
        self.max_pos_items = max_pos_items
        self.num_workers = num_workers
        self.seed = seed

        self.num_items = 0
        self.n_markets = 0
        self.n_categories = 0
        self.text_embeddings = {}
        self.item_markets = None  # np.array (num_items+1,) of market indices
        self.item_categories = None  # np.array (num_items+1,) of category indices
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "complement_preprocessed.pkl").exists():
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

    # --- Preprocessing Pipeline ---

    def _run_preprocessing(self):
        # 1. Load raw interactions (with raw event types preserved for deny filter)
        df = self._load_interactions()
        logger.info(f"Loaded {len(df)} raw interactions")

        # 2. Filter to on-sale items
        on_sale_items = self._get_on_sale_items()
        df = df[df["ITEM_ID"].isin(on_sale_items)]
        logger.info(f"After on-sale filter: {len(df)} interactions")

        # 3. Filter users with min actions
        df = self._filter_short_users(df, count_unique=False)
        logger.info(f"After min-user filter: {len(df)} interactions")

        # 4. Apply deny condition (clickbait filter)
        denied_items = self._find_deny_items(df)
        if denied_items:
            logger.info(f"Deny condition filtered {len(denied_items)} items")
            df = df[~df["ITEM_ID"].isin(denied_items)]

        # 5. Dual vocabulary: interaction items (top N) + order/cart items (top M)
        on_voca_interaction = self._get_top_items(df, self.max_interaction_items)
        logger.info(f"Interaction vocabulary: {len(on_voca_interaction)} items")

        order_cart_df = df[df["EVENT_TYPE"].isin({"order", "cart"})]
        on_voca_order = self._get_top_items(order_cart_df, self.max_ordered_items)
        logger.info(f"Order/cart vocabulary: {len(on_voca_order)} items")

        total_on_voca: Set[int] = set(on_voca_interaction) | set(on_voca_order)
        logger.info(f"Total vocabulary: {len(total_on_voca)} items")

        df = df[df["ITEM_ID"].isin(total_on_voca)]
        logger.info(f"After vocab filter: {len(df)} interactions")

        # 6. Re-filter users after vocab reduction
        df = self._filter_short_users(df, count_unique=True)
        logger.info(f"After re-filter: {len(df)} interactions")

        # 7. Apply final event mapping: order/cart/like -> preference
        df["EVENT_TYPE"] = df["EVENT_TYPE"].map(FINAL_EVENT_MAP).fillna("unknown")
        df = df[df["EVENT_TYPE"] != "unknown"]

        # 8. Create item encoder (item_id -> 1-based index)
        item_ids = df["ITEM_ID"].value_counts().index.tolist()
        self.item_id2index = {str(iid): idx + 1 for idx, iid in enumerate(item_ids)}
        self.item_id2index["unknown"] = 0
        self.num_items = len(item_ids)
        logger.info(f"Vocabulary size: {self.num_items}")

        # 9. Map IDs to indices
        df["item_index"] = df["ITEM_ID"].astype(str).map(self.item_id2index)
        df = df.dropna(subset=["item_index"])
        df["item_index"] = df["item_index"].astype(int)

        # 10. Build market / category label encoders and per-item features
        self._build_item_features(df, total_on_voca)

        # 11. Split by users into train/test sequences
        train_seqs, test_seqs = self._split_by_users(df)
        logger.info(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

        # 12. Generate text embeddings
        self.text_embeddings = self._generate_text_embeddings(total_on_voca)
        logger.info(f"Generated text embeddings for {len(self.text_embeddings)} items")

        # 13. Create datasets
        rng = random.Random(self.seed)
        self.train_dataset = ComplementTrainDataset(
            train_seqs,
            self.max_len,
            self.num_items,
            rng,
            max_pos_items=self.max_pos_items,
            negative_sampling=self.negative_sampling,
        )
        self.val_dataset = ComplementEvalDataset(test_seqs, self.max_len)

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

        # Load order data
        if self.order_path and os.path.exists(self.order_path):
            logger.info("Loading order data...")
            df_order = pd.read_csv(self.order_path)
            df_order = df_order.rename(columns={"goods_sno": "ITEM_ID"})
            df_order["USER_ID"] = df_order["member_sno"].apply(lambda x: f"m{x}")
            df_order["TIMESTAMP"] = df_order["order_sno"] / 1000
            df_order["EVENT_TYPE"] = "order"
            dfs.append(df_order[["USER_ID", "ITEM_ID", "TIMESTAMP", "EVENT_TYPE"]])

        df = pd.concat(dfs, ignore_index=True)

        # Apply raw event mapping (preserves order/cart/like as separate types for deny filter)
        df["EVENT_TYPE"] = df["EVENT_TYPE"].map(RAW_EVENT_MAP).fillna("unknown")
        df = df[df["EVENT_TYPE"] != "unknown"]
        return df

    def _get_on_sale_items(self) -> set:
        df_goods = pd.read_csv(self.goods_path, usecols=["sno"], escapechar="\\")
        return set(df_goods["sno"].tolist())

    # --- Deny Condition Filter ---

    def _find_deny_items(self, df: pd.DataFrame) -> List[int]:
        """Find clickbait items: high clicks but very low preference events ratio.

        Items with clicks >= deny_condition_min_click AND
        (order+cart+like)/clicks < deny_condition_events_per_click are excluded.
        """
        target_events = ["order", "cart", "like"]

        try:
            stat = (
                df.groupby(["ITEM_ID", "EVENT_TYPE"])
                .agg(cnt=("USER_ID", "count"))
                .reset_index()
                .pivot(index="ITEM_ID", columns="EVENT_TYPE", values="cnt")
                .fillna(0)
                .reset_index()
            )

            # Ensure required columns exist
            for col in target_events + ["click"]:
                if col not in stat.columns:
                    stat[col] = 0

            stat["events_per_click"] = stat[target_events].sum(axis=1) / (1.0 + stat["click"])

            denied = stat[
                (stat["click"] >= self.deny_condition_min_click)
                & (stat["events_per_click"] < self.deny_condition_events_per_click)
            ]
            return denied["ITEM_ID"].tolist()
        except Exception:
            return []

    # --- Filtering ---

    def _filter_short_users(self, df, count_unique=False):
        if count_unique:
            counts = df.groupby("USER_ID")["ITEM_ID"].nunique()
        else:
            counts = df.groupby("USER_ID").size()
        good_users = counts[counts >= self.min_actions_per_user].index
        return df[df["USER_ID"].isin(good_users)]

    def _get_top_items(self, df, max_items) -> list:
        item_counts = df.groupby("ITEM_ID").size().reset_index(name="count")
        item_counts = item_counts.sort_values("count", ascending=False)
        if max_items and len(item_counts) > max_items:
            item_counts = item_counts.head(max_items)
        return item_counts["ITEM_ID"].tolist()

    # --- Item Features (Market / Category) ---

    def _build_item_features(self, df, on_voca_items):
        """Build per-item market and category index arrays."""
        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

        # Market label encoding
        if "market_sno" in df_goods.columns:
            market_ids = sorted(df_goods["market_sno"].dropna().unique().tolist())
            self.market_id2index = {mid: idx + 1 for idx, mid in enumerate(market_ids)}
            self.n_markets = len(market_ids)
        else:
            self.market_id2index = {}
            self.n_markets = 0

        # Category label encoding (using category_path for complement, not standard_category)
        df_category = pd.read_csv(self.category_path)
        if "sno" in df_category.columns:
            cat_ids = sorted(df_category["sno"].dropna().unique().tolist())
            self.category_id2index = {cid: idx + 1 for idx, cid in enumerate(cat_ids)}
            self.n_categories = len(cat_ids)
        else:
            self.category_id2index = {}
            self.n_categories = 0

        # Build per-item arrays (index 0 = padding/unknown)
        self.item_markets = np.zeros(self.num_items + 1, dtype=np.int64)
        self.item_categories = np.zeros(self.num_items + 1, dtype=np.int64)

        df_goods["item_index"] = df_goods["sno"].astype(str).map(self.item_id2index)
        df_goods = df_goods.dropna(subset=["item_index"])
        df_goods["item_index"] = df_goods["item_index"].astype(int)

        for _, row in df_goods.iterrows():
            idx = int(row["item_index"])
            if self.n_markets > 0 and "market_sno" in row and pd.notna(row["market_sno"]):
                self.item_markets[idx] = self.market_id2index.get(row["market_sno"], 0)
            if self.n_categories > 0 and "category_sno" in row and pd.notna(row.get("category_sno")):
                self.item_categories[idx] = self.category_id2index.get(row["category_sno"], 0)

    # --- Splitting ---

    def _split_by_users(self, df):
        rng = random.Random(self.seed)
        train_sequences = []
        test_sequences = []

        for _, group in df.groupby("USER_ID"):
            group = group.sort_values("TIMESTAMP")
            items = group["item_index"].values
            # Deduplicate while preserving order
            _, unique_idx = np.unique(items, return_index=True)
            unique_idx = np.sort(unique_idx)
            items = items[unique_idx].tolist()

            if len(items) < 2:
                continue

            # Pick test items (up to num_test_samples_per_user)
            if len(test_sequences) < self.max_test_samples and len(items) > 2:
                n_test = min(self.num_test_samples_per_user, len(items) // 2)
                test_item_indices = sorted(rng.sample(range(len(items)), n_test), reverse=True)
                answer_items = [items[i] for i in test_item_indices]
                train_items = [items[i] for i in range(len(items)) if i not in set(test_item_indices)]

                test_sequences.append({
                    "items": train_items,
                    "answer_items": answer_items,
                })
            else:
                train_items = items

            train_sequences.append({
                "items": train_items,
            })

        return train_sequences, test_sequences

    # --- Text Embeddings ---

    def _generate_text_embeddings(self, on_voca_items) -> dict:
        from sentence_transformers import SentenceTransformer

        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

        df_category = pd.read_csv(self.category_path)
        cat_map = df_category.set_index("sno")["catnm"].astype(str).to_dict()

        df_goods["category_name"] = df_goods["category_sno"].map(cat_map).fillna("")
        df_goods["item_text"] = df_goods["category_name"] + " " + df_goods["name"].astype(str)

        df_goods["item_index"] = df_goods["sno"].astype(str).map(self.item_id2index)
        df_goods = df_goods.dropna(subset=["item_index"])
        df_goods["item_index"] = df_goods["item_index"].astype(int)

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
            "n_markets": self.n_markets,
            "n_categories": self.n_categories,
            "item_id2index": self.item_id2index,
            "text_embeddings": self.text_embeddings,
            "item_markets": self.item_markets,
            "item_categories": self.item_categories,
            "train_sequences": self.train_dataset.sequences,
            "test_sequences": self.val_dataset.sequences,
        }
        with open(cache_path / "complement_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached preprocessed data to {cache_path / 'complement_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "complement_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_items = data["num_items"]
        self.n_markets = data["n_markets"]
        self.n_categories = data["n_categories"]
        self.item_id2index = data["item_id2index"]
        self.text_embeddings = data["text_embeddings"]
        self.item_markets = data["item_markets"]
        self.item_categories = data["item_categories"]

        rng = random.Random(self.seed)
        self.train_dataset = ComplementTrainDataset(
            data["train_sequences"],
            self.max_len,
            self.num_items,
            rng,
            max_pos_items=self.max_pos_items,
            negative_sampling=self.negative_sampling,
        )
        self.val_dataset = ComplementEvalDataset(data["test_sequences"], self.max_len)


def _insert_pad(seq, max_len):
    """Left-pad a sequence with zeros and truncate to max_len."""
    pad_len = max_len - len(seq)
    seq_padded = [0] * pad_len + seq
    seq_padded = seq_padded[-max_len:]
    return seq_padded
