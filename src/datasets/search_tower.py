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

SECONDS_PER_WEEK = 7 * 24 * 3600


# --- Datasets ---


class SearchTowerTrainDataset(data_utils.Dataset):
    """Training dataset for search tower.

    Each sample is a search query with:
        - click history before the query (item indices)
        - query tokens
        - normalized query timestamp
        - multi-hot labels: clicked items + augmented same-category items
    """

    def __init__(self, samples, max_len, num_items, category_map, rng, use_augment=True):
        self.samples = samples
        self.max_len = max_len
        self.num_items = num_items
        self.category_map = category_map  # item_index -> category_index
        self.rng = rng
        self.use_augment = use_augment

        # Build reverse map: category_index -> list of item_indices
        self.category_items = {}
        if use_augment and category_map:
            for item_idx, cat_idx in category_map.items():
                if cat_idx not in self.category_items:
                    self.category_items[cat_idx] = []
                self.category_items[cat_idx].append(item_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        click_items = list(sample["click_items"])
        query_tokens = list(sample["query_tokens"])
        query_time = sample["query_time"]
        positive_items = list(sample["positive_items"])

        # Truncate / pad click history to max_len (left-padded)
        if len(click_items) > self.max_len:
            click_items = click_items[-self.max_len :]
        pad_len = self.max_len - len(click_items)
        click_items = [0] * pad_len + click_items

        # Build multi-hot target vector (num_items+1, index 0 = padding)
        targets = torch.zeros(self.num_items + 1)
        for item_idx in positive_items:
            if 0 < item_idx <= self.num_items:
                targets[item_idx] = 1.0

        # Augment: add same-category items as positives
        if self.use_augment and self.category_map:
            for item_idx in positive_items:
                cat_idx = self.category_map.get(item_idx, None)
                if cat_idx is not None and cat_idx in self.category_items:
                    for sibling in self.category_items[cat_idx]:
                        if 0 < sibling <= self.num_items:
                            targets[sibling] = 1.0

        # Normalize targets so they sum to 1 (probability distribution)
        target_sum = targets[1:].sum()
        if target_sum > 0:
            targets[1:] = targets[1:] / target_sum

        # Pad query tokens
        max_query_len = 32
        if len(query_tokens) > max_query_len:
            query_tokens = query_tokens[:max_query_len]
        q_pad_len = max_query_len - len(query_tokens)
        query_tokens = query_tokens + [0] * q_pad_len

        return (
            {
                "click_items": torch.LongTensor(click_items),
                "query_tokens": torch.LongTensor(query_tokens),
                "query_time": torch.FloatTensor([query_time]).squeeze(0),
            },
            targets,
        )


class SearchTowerEvalDataset(data_utils.Dataset):
    """Evaluation dataset for search tower.

    Each sample returns the first positive item as a single target.
    """

    def __init__(self, samples, max_len, num_items):
        self.samples = samples
        self.max_len = max_len
        self.num_items = num_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        click_items = list(sample["click_items"])
        query_tokens = list(sample["query_tokens"])
        query_time = sample["query_time"]
        positive_items = sample["positive_items"]

        # Truncate / pad click history
        if len(click_items) > self.max_len:
            click_items = click_items[-self.max_len :]
        pad_len = self.max_len - len(click_items)
        click_items = [0] * pad_len + click_items

        # Pad query tokens
        max_query_len = 32
        if len(query_tokens) > max_query_len:
            query_tokens = query_tokens[:max_query_len]
        q_pad_len = max_query_len - len(query_tokens)
        query_tokens = query_tokens + [0] * q_pad_len

        # Target: first positive item (for NDCG / Recall computation)
        target = torch.LongTensor(positive_items[:1])

        return (
            {
                "click_items": torch.LongTensor(click_items),
                "query_tokens": torch.LongTensor(query_tokens),
                "query_time": torch.FloatTensor([query_time]).squeeze(0),
            },
            target,
        )


# --- DataModule ---


class SearchTowerDataModule(L.LightningDataModule):
    """Search Tower DataModule.

    Handles the full pipeline: loading interaction + search query data,
    merging search queries with click history, filtering, label encoding,
    text embedding generation, train/test splitting, and dataset creation.
    """

    def __init__(
        self,
        interaction_dir: str,
        search_dir: str,
        goods_path: str,
        category_path: str,
        standard_category_path: str,
        pretrained_text_path: str,
        cache_dir: str = "",
        max_len: int = 80,
        batch_size: int = 512,
        max_items: int = 180000,
        min_actions_per_user: int = 100,
        max_test_samples: int = 200000,
        num_workers: int = 4,
        use_augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.search_dir = search_dir
        self.goods_path = goods_path
        self.category_path = category_path
        self.standard_category_path = standard_category_path
        self.pretrained_text_path = pretrained_text_path
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_items = max_items
        self.min_actions_per_user = min_actions_per_user
        self.max_test_samples = max_test_samples
        self.num_workers = num_workers
        self.use_augment = use_augment
        self.seed = seed

        self.num_items = 0
        self.text_embeddings = {}
        self.category_map = {}
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "search_tower_preprocessed.pkl").exists():
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
        # 1. Load raw interactions (click history)
        df_interactions = self._load_interactions()
        logger.info(f"Loaded {len(df_interactions)} raw interactions")

        # 2. Load search queries
        df_search = self._load_search_queries()
        logger.info(f"Loaded {len(df_search)} search queries")

        # 3. Filter to on-sale items
        on_sale_items = self._get_on_sale_items()
        df_interactions = df_interactions[df_interactions["ITEM_ID"].isin(on_sale_items)]
        logger.info(f"After on-sale filter: {len(df_interactions)} interactions")

        # 4. Filter users with min actions
        df_interactions = self._filter_short_users(df_interactions)
        logger.info(f"After min-user filter: {len(df_interactions)} interactions")

        # 5. Keep top max_items by popularity
        on_voca_items = self._get_top_items(df_interactions)
        df_interactions = df_interactions[df_interactions["ITEM_ID"].isin(on_voca_items)]
        logger.info(f"After vocab filter: {len(df_interactions)} interactions, {len(on_voca_items)} items")

        # 6. Create item encoder (item_id -> 1-based index)
        item_ids = df_interactions["ITEM_ID"].value_counts().index.tolist()
        self.item_id2index = {str(iid): idx + 1 for idx, iid in enumerate(item_ids)}
        self.item_id2index["unknown"] = 0
        self.num_items = len(item_ids)
        logger.info(f"Vocabulary size: {self.num_items}")

        # 7. Map IDs to indices
        df_interactions["item_index"] = df_interactions["ITEM_ID"].astype(str).map(self.item_id2index)
        df_interactions = df_interactions.dropna(subset=["item_index"])
        df_interactions["item_index"] = df_interactions["item_index"].astype(int)

        # 8. Build category map (item_index -> category)
        self.category_map = self._build_category_map(on_voca_items)
        logger.info(f"Category map: {len(self.category_map)} items mapped")

        # 9. Merge search queries with click history
        train_samples, test_samples = self._build_query_click_pairs(df_interactions, df_search)
        logger.info(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

        # 10. Generate text embeddings
        self.text_embeddings = self._generate_text_embeddings(on_voca_items)
        logger.info(f"Generated text embeddings for {len(self.text_embeddings)} items")

        # 11. Create datasets
        rng = random.Random(self.seed)
        self.train_dataset = SearchTowerTrainDataset(
            train_samples, self.max_len, self.num_items, self.category_map, rng, self.use_augment
        )
        self.val_dataset = SearchTowerEvalDataset(test_samples, self.max_len, self.num_items)

    # --- Data Loading ---

    def _load_interactions(self) -> pd.DataFrame:
        csv_files = sorted(glob(os.path.join(self.interaction_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.interaction_dir}")
        logger.info(f"Loading {len(csv_files)} interaction CSV files...")

        # Detect available columns from first file
        sample_df = pd.read_csv(csv_files[0], nrows=0)
        available_cols = set(sample_df.columns)

        # Determine columns to load
        required_cols = []
        col_map = {}
        for target, alternatives in [
            ("USER_ID", ["USER_ID"]),
            ("ITEM_ID", ["ITEM_ID"]),
            ("TIMESTAMP", ["TIMESTAMP"]),
            ("EVENT_TYPE", ["EVENT_TYPE", "event_name"]),
        ]:
            for alt in alternatives:
                if alt in available_cols:
                    required_cols.append(alt)
                    if alt != target:
                        col_map[alt] = target
                    break

        optional_cols = ["SEARCH_QUERY_ID", "screen_name"]
        for col in optional_cols:
            if col in available_cols:
                required_cols.append(col)

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f, usecols=required_cols)
            dfs.append(df)
        result = pd.concat(dfs, ignore_index=True)

        # Rename columns to standard names
        if col_map:
            result = result.rename(columns=col_map)

        # If screen_name is available but SEARCH_QUERY_ID is not, derive search queries
        if "screen_name" in result.columns and "SEARCH_QUERY_ID" not in result.columns:
            # Use screen_name=SEARCH_RESULT to identify search-related interactions
            search_mask = result["screen_name"].str.contains("SEARCH", case=False, na=False)
            # Generate synthetic query IDs by grouping consecutive search interactions per user
            result["SEARCH_QUERY_ID"] = np.nan
            result.loc[search_mask, "SEARCH_QUERY_ID"] = (
                result.loc[search_mask]
                .groupby("USER_ID")["TIMESTAMP"]
                .transform(lambda ts: (ts.diff().fillna(0) > 60).cumsum().astype(str))
            )
            # Make query IDs unique per user
            search_rows = result["SEARCH_QUERY_ID"].notna()
            result.loc[search_rows, "SEARCH_QUERY_ID"] = (
                result.loc[search_rows, "USER_ID"].astype(str) + "_q" + result.loc[search_rows, "SEARCH_QUERY_ID"]
            )

        return result

    def _load_search_queries(self) -> pd.DataFrame:
        """Load search query AVRO files from search_dir."""
        try:
            import fastavro
        except ImportError:
            fastavro = None

        avro_files = sorted(glob(os.path.join(self.search_dir, "*.avro")))
        if not avro_files:
            # Fallback to CSV
            csv_files = sorted(glob(os.path.join(self.search_dir, "*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"No AVRO or CSV files found in {self.search_dir}")
            logger.info(f"Loading {len(csv_files)} search CSV files...")
            dfs = []
            for f in csv_files:
                df = pd.read_csv(f)
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            logger.info(f"Loading {len(avro_files)} search AVRO files...")
            records = []
            for f in avro_files:
                with open(f, "rb") as fp:
                    reader = fastavro.reader(fp)
                    for record in reader:
                        records.append(record)
            df = pd.DataFrame(records)

        # Normalize column names
        col_map = {}
        for target, alternatives in [
            ("USER_ID", ["USER_ID", "user_id"]),
            ("TIMESTAMP", ["TIMESTAMP", "timestamp"]),
            ("QUERY", ["QUERY", "query", "QUERY_TEXT", "query_text"]),
            ("QUERY_TOKENS", ["QUERY_TOKENS", "query_tokens"]),
        ]:
            for alt in alternatives:
                if alt in df.columns:
                    if alt != target:
                        col_map[alt] = target
                    break
        if col_map:
            df = df.rename(columns=col_map)

        # Ensure TIMESTAMP is numeric
        if "TIMESTAMP" in df.columns:
            df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")

        return df

    def _get_on_sale_items(self) -> set:
        df_goods = pd.read_csv(self.goods_path, usecols=["sno"], escapechar="\\")
        return set(df_goods["sno"].tolist())

    # --- Filtering ---

    def _filter_short_users(self, df):
        counts = df.groupby("USER_ID").size()
        good_users = counts[counts >= self.min_actions_per_user].index
        return df[df["USER_ID"].isin(good_users)]

    def _get_top_items(self, df) -> list:
        item_counts = df.groupby("ITEM_ID").size().reset_index(name="count")
        item_counts = item_counts.sort_values("count", ascending=False)
        if self.max_items and len(item_counts) > self.max_items:
            item_counts = item_counts.head(self.max_items)
        return item_counts["ITEM_ID"].tolist()

    # --- Category Map ---

    def _build_category_map(self, on_voca_items) -> dict:
        """Build item_index -> category_index mapping for augmentation."""
        df_goods = pd.read_csv(self.goods_path, escapechar="\\")
        df_goods = df_goods.drop_duplicates(subset=["sno"])
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

        # Use standard_category_sno from goods if available, else fall back to category_sno
        if "standard_category_sno" in df_goods.columns:
            cat_series = df_goods.set_index("sno")["standard_category_sno"]
        elif "category_sno" in df_goods.columns:
            cat_series = df_goods.set_index("sno")["category_sno"]
        else:
            return {}

        category_map = {}
        for item_id, cat_val in cat_series.items():
            str_id = str(item_id)
            if str_id in self.item_id2index and pd.notna(cat_val):
                item_index = self.item_id2index[str_id]
                category_map[item_index] = int(cat_val)
        return category_map

    # --- Query-Click Pair Building ---

    @staticmethod
    def _tokenize_query(query_text, max_vocab=29999):
        """Simple hash-based tokenizer for search query text.

        Maps each character-bigram to an integer in [1, max_vocab].
        """
        if not isinstance(query_text, str) or not query_text.strip():
            return []
        text = query_text.strip()
        tokens = []
        # Character bigram hashing (works for Korean/multilingual text)
        for i in range(len(text)):
            bigram = text[i:i+2] if i + 1 < len(text) else text[i]
            token_id = (hash(bigram) % max_vocab) + 1
            tokens.append(token_id)
        return tokens[:32]  # max_query_len

    def _build_query_click_pairs(self, df_interactions, df_search):
        """Merge search queries with user click history.

        Supports two modes:
        1. SEARCH_QUERY_ID-based: direct join when interactions have query IDs
        2. Timestamp-based: match search queries to search-result clicks by user+time proximity
        """
        rng = random.Random(self.seed)
        train_samples = []
        test_samples = []

        has_query_id = "SEARCH_QUERY_ID" in df_interactions.columns and df_interactions["SEARCH_QUERY_ID"].notna().any()
        has_query_text = "QUERY" in df_search.columns
        has_query_tokens = "QUERY_TOKENS" in df_search.columns

        # Sort interactions by timestamp
        df_interactions = df_interactions.sort_values(["USER_ID", "TIMESTAMP"])

        if has_query_id and has_query_tokens:
            # Mode 1: SEARCH_QUERY_ID-based join (original approach)
            return self._build_pairs_by_query_id(df_interactions, df_search, rng)

        if has_query_text:
            # Mode 2: Timestamp-based matching with raw query text
            return self._build_pairs_by_timestamp(df_interactions, df_search, rng)

        logger.warning("No usable search query data found.")
        return train_samples, test_samples

    def _build_pairs_by_query_id(self, df_interactions, df_search, rng):
        """Build pairs using SEARCH_QUERY_ID column to join."""
        train_samples = []
        test_samples = []

        df_search_clicks = df_interactions[df_interactions["SEARCH_QUERY_ID"].notna()].copy()
        if len(df_search_clicks) == 0:
            return train_samples, test_samples

        df_search_clicks = df_search_clicks.merge(df_search, on="SEARCH_QUERY_ID", how="inner", suffixes=("", "_query"))
        if len(df_search_clicks) == 0:
            return train_samples, test_samples

        query_token_col = "QUERY_TOKENS" if "QUERY_TOKENS" in df_search_clicks.columns else "query_tokens"

        for (user_id, query_id), group in df_search_clicks.groupby(["USER_ID", "SEARCH_QUERY_ID"]):
            query_time_raw = group["TIMESTAMP"].min()
            query_time = float(np.clip(query_time_raw / (SECONDS_PER_WEEK * 52), 0.0, 1.0))

            tokens_raw = group[query_token_col].iloc[0] if query_token_col in group.columns else None
            if isinstance(tokens_raw, str):
                query_tokens = [int(t) for t in tokens_raw.split(",") if t.strip().isdigit()]
            elif isinstance(tokens_raw, (list, np.ndarray)):
                query_tokens = [int(t) for t in tokens_raw]
            else:
                query_tokens = []
            if not query_tokens:
                continue

            positive_items = [self.item_id2index[str(iid)] for iid in group["ITEM_ID"].unique() if str(iid) in self.item_id2index]
            if not positive_items:
                continue

            user_history = df_interactions[(df_interactions["USER_ID"] == user_id) & (df_interactions["TIMESTAMP"] < query_time_raw)]
            click_items = user_history["item_index"].dropna().astype(int).tolist()
            if len(click_items) < 2:
                continue

            sample = {"click_items": click_items, "query_tokens": query_tokens, "query_time": query_time, "positive_items": positive_items}
            if len(test_samples) < self.max_test_samples and rng.random() < 0.1:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

        return train_samples, test_samples

    def _build_pairs_by_timestamp(self, df_interactions, df_search, rng):
        """Build pairs by matching search queries to interactions by user + timestamp proximity."""
        train_samples = []
        test_samples = []

        # Identify search-result clicks
        if "screen_name" in df_interactions.columns:
            search_click_mask = df_interactions["screen_name"].str.contains("SEARCH", case=False, na=False)
        else:
            search_click_mask = pd.Series(False, index=df_interactions.index)

        df_search_clicks = df_interactions[search_click_mask].copy()
        if len(df_search_clicks) == 0:
            logger.warning("No search-result clicks found in interactions.")
            return train_samples, test_samples

        # Normalize search query columns
        df_search = df_search.copy()

        # Get users who have both search queries and search clicks
        search_users = set(df_search["USER_ID"].unique()) & set(df_search_clicks["USER_ID"].unique())
        logger.info(f"Users with both search queries and search clicks: {len(search_users)}")

        # Window (seconds) to match a search query to clicks
        match_window = 300  # 5 minutes

        for user_id in search_users:
            user_queries = df_search[df_search["USER_ID"] == user_id].sort_values("TIMESTAMP")
            user_search_clicks = df_search_clicks[df_search_clicks["USER_ID"] == user_id].sort_values("TIMESTAMP")
            user_all = df_interactions[df_interactions["USER_ID"] == user_id].sort_values("TIMESTAMP")

            for _, query_row in user_queries.iterrows():
                query_time_raw = query_row["TIMESTAMP"]
                query_text = query_row.get("QUERY", "")

                query_tokens = self._tokenize_query(query_text)
                if not query_tokens:
                    continue

                # Find search clicks within the match window after the query
                matched_clicks = user_search_clicks[
                    (user_search_clicks["TIMESTAMP"] >= query_time_raw) &
                    (user_search_clicks["TIMESTAMP"] <= query_time_raw + match_window)
                ]
                if len(matched_clicks) == 0:
                    continue

                positive_items = []
                for iid in matched_clicks["ITEM_ID"].unique():
                    str_id = str(iid)
                    if str_id in self.item_id2index:
                        positive_items.append(self.item_id2index[str_id])
                if not positive_items:
                    continue

                # Click history before the query
                history = user_all[user_all["TIMESTAMP"] < query_time_raw]
                click_items = history["item_index"].dropna().astype(int).tolist()
                if len(click_items) < 2:
                    continue

                query_time = float(np.clip(query_time_raw / (SECONDS_PER_WEEK * 52), 0.0, 1.0))

                sample = {
                    "click_items": click_items,
                    "query_tokens": query_tokens,
                    "query_time": query_time,
                    "positive_items": positive_items,
                }

                if len(test_samples) < self.max_test_samples and rng.random() < 0.1:
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

            if len(train_samples) + len(test_samples) > 500000:
                break  # Sufficient samples

        return train_samples, test_samples

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
            "item_id2index": self.item_id2index,
            "text_embeddings": self.text_embeddings,
            "category_map": self.category_map,
            "train_samples": self.train_dataset.samples,
            "test_samples": self.val_dataset.samples,
        }
        with open(cache_path / "search_tower_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached preprocessed data to {cache_path / 'search_tower_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "search_tower_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_items = data["num_items"]
        self.item_id2index = data["item_id2index"]
        self.text_embeddings = data["text_embeddings"]
        self.category_map = data["category_map"]

        rng = random.Random(self.seed)
        self.train_dataset = SearchTowerTrainDataset(
            data["train_samples"], self.max_len, self.num_items, self.category_map, rng, self.use_augment
        )
        self.val_dataset = SearchTowerEvalDataset(data["test_samples"], self.max_len, self.num_items)
