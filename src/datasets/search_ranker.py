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

# Label constants
NEUTRAL = -1

# Max items per session for candidate set
DEFAULT_MAX_SESSION_ITEMS = 100
DEFAULT_SEQ_MAX_LEN = 200
DEFAULT_QUERY_MAX_LEN = 20


class SearchRankerDataset(data_utils.Dataset):
    """Per-session dataset for search ranker training/evaluation.

    Each sample represents one search session with:
    - User click history (item IDs, market IDs, category IDs)
    - Last search query tokens
    - Candidate items for the session (up to max_session_items)
    - Per-candidate labels for each task: retriever (click), ctr, ctcar, ctcvr
    """

    def __init__(
        self,
        sessions: list[dict],
        seq_max_len: int = DEFAULT_SEQ_MAX_LEN,
        max_session_items: int = DEFAULT_MAX_SESSION_ITEMS,
        query_max_len: int = DEFAULT_QUERY_MAX_LEN,
    ):
        self.sessions = sessions
        self.seq_max_len = seq_max_len
        self.max_session_items = max_session_items
        self.query_max_len = query_max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, index):
        session = self.sessions[index]

        # --- Click history (user side) ---
        click_items = session["click_items"]
        click_markets = session["click_markets"]
        click_categories = session["click_categories"]

        # Truncate to seq_max_len (keep most recent)
        click_items = click_items[-self.seq_max_len:]
        click_markets = click_markets[-self.seq_max_len:]
        click_categories = click_categories[-self.seq_max_len:]

        # Left-pad to seq_max_len
        pad_len = self.seq_max_len - len(click_items)
        click_items = [0] * pad_len + click_items
        click_markets = [0] * pad_len + click_markets
        click_categories = [0] * pad_len + click_categories

        # --- Query tokens ---
        query_tokens = session["query_tokens"]
        query_tokens = query_tokens[:self.query_max_len]
        query_pad = self.query_max_len - len(query_tokens)
        query_tokens = query_tokens + [0] * query_pad

        # --- Candidate items and labels ---
        candidate_items = session["candidate_items"]
        label_retriever = session["label_retriever"]
        label_ctr = session["label_ctr"]
        label_ctcar = session["label_ctcar"]
        label_ctcvr = session["label_ctcvr"]

        # Truncate/pad candidates to max_session_items
        num_candidates = len(candidate_items)
        if num_candidates > self.max_session_items:
            candidate_items = candidate_items[:self.max_session_items]
            label_retriever = label_retriever[:self.max_session_items]
            label_ctr = label_ctr[:self.max_session_items]
            label_ctcar = label_ctcar[:self.max_session_items]
            label_ctcvr = label_ctcvr[:self.max_session_items]
        elif num_candidates < self.max_session_items:
            pad_n = self.max_session_items - num_candidates
            candidate_items = candidate_items + [0] * pad_n
            label_retriever = label_retriever + [0] * pad_n
            label_ctr = label_ctr + [NEUTRAL] * pad_n
            label_ctcar = label_ctcar + [NEUTRAL] * pad_n
            label_ctcvr = label_ctcvr + [NEUTRAL] * pad_n

        inputs = {
            "click_items": torch.LongTensor(click_items),
            "click_markets": torch.LongTensor(click_markets),
            "click_categories": torch.LongTensor(click_categories),
            "query_tokens": torch.LongTensor(query_tokens),
            "candidate_items": torch.LongTensor(candidate_items),
        }

        labels = {
            "retriever": torch.FloatTensor(label_retriever),
            "ctr": torch.FloatTensor(label_ctr),
            "ctcar": torch.FloatTensor(label_ctcar),
            "ctcvr": torch.FloatTensor(label_ctcvr),
        }

        return inputs, labels


class SearchRankerDataModule(L.LightningDataModule):
    """Search Ranker DataModule.

    Loads:
    - Labels: parquet with search exposure/click labels + CTR/CTCAR/CTCVR labels
    - Click history: CSV (user interaction history)
    - Search history: parquet (search queries per session)

    Each sample is a search session with up to max_session_items candidates,
    user click history, and last search query tokens.
    """

    def __init__(
        self,
        labels_path: str,
        click_history_dir: str,
        search_history_path: str,
        seq_max_len: int = DEFAULT_SEQ_MAX_LEN,
        max_session_items: int = DEFAULT_MAX_SESSION_ITEMS,
        query_max_len: int = DEFAULT_QUERY_MAX_LEN,
        batch_size: int = 512,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.labels_path = labels_path
        self.click_history_dir = click_history_dir
        self.search_history_path = search_history_path
        self.seq_max_len = seq_max_len
        self.max_session_items = max_session_items
        self.query_max_len = query_max_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        logger.info("Loading search ranker data...")

        # 1. Load labels
        df_labels = self._load_labels()
        logger.info(f"Loaded {len(df_labels)} label rows")

        # 2. Load click history
        click_history = self._load_click_history()
        logger.info(f"Loaded click history for {len(click_history)} users")

        # 3. Load search history
        search_history = self._load_search_history()
        logger.info(f"Loaded search history for {len(search_history)} sessions")

        # 4. Build sessions
        train_sessions, test_sessions = self._build_sessions(df_labels, click_history, search_history)
        logger.info(f"Train sessions: {len(train_sessions)}, Test sessions: {len(test_sessions)}")

        # 5. Create datasets
        self.train_dataset = SearchRankerDataset(
            train_sessions,
            seq_max_len=self.seq_max_len,
            max_session_items=self.max_session_items,
            query_max_len=self.query_max_len,
        )
        self.test_dataset = SearchRankerDataset(
            test_sessions,
            seq_max_len=self.seq_max_len,
            max_session_items=self.max_session_items,
            query_max_len=self.query_max_len,
        )

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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    # --- Data Loading ---

    def _load_labels(self) -> pd.DataFrame:
        """Load label parquet with columns:
        session_id, user_id, item_id, market_id, category_id,
        is_clicked, is_ctr, is_ctcar, is_ctcvr, is_train
        """
        df = pd.read_parquet(self.labels_path)
        return df

    def _load_click_history(self) -> dict:
        """Load user click history CSVs into a dict: user_id -> list of dicts.

        Each dict has keys: item_id, market_id, category_id, timestamp.
        Sorted by timestamp ascending.
        """
        csv_files = sorted(glob(os.path.join(self.click_history_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.click_history_dir}")

        logger.info(f"Loading {len(csv_files)} click history CSV files...")
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values("timestamp")

        history = {}
        for user_id, group in df_all.groupby("user_id"):
            history[user_id] = {
                "item_ids": group["item_id"].tolist(),
                "market_ids": group["market_id"].tolist(),
                "category_ids": group["category_id"].tolist(),
            }
        return history

    def _load_search_history(self) -> dict:
        """Load search history parquet into dict: session_id -> query_tokens (list of int)."""
        df = pd.read_parquet(self.search_history_path)
        search_hist = {}
        for _, row in df.iterrows():
            tokens = row["query_tokens"]
            if isinstance(tokens, str):
                tokens = [int(t) for t in tokens.split(",") if t.strip()]
            elif isinstance(tokens, (list, np.ndarray)):
                tokens = [int(t) for t in tokens]
            else:
                tokens = []
            search_hist[row["session_id"]] = tokens
        return search_hist

    # --- Session Building ---

    def _build_sessions(self, df_labels, click_history, search_history):
        """Group label rows by session, attach click history and query tokens.

        Returns:
            train_sessions, test_sessions: lists of session dicts
        """
        train_sessions = []
        test_sessions = []

        for session_id, group in df_labels.groupby("session_id"):
            user_id = group["user_id"].iloc[0]
            is_train = group["is_train"].iloc[0]

            # Click history for this user
            user_hist = click_history.get(user_id, {"item_ids": [], "market_ids": [], "category_ids": []})

            # Query tokens for this session
            query_tokens = search_history.get(session_id, [])

            # Candidate items and labels
            candidate_items = group["item_id"].tolist()
            label_retriever = group["is_clicked"].astype(float).tolist()
            label_ctr = group["is_ctr"].astype(float).tolist()
            label_ctcar = group["is_ctcar"].astype(float).tolist()
            label_ctcvr = group["is_ctcvr"].astype(float).tolist()

            session = {
                "click_items": user_hist["item_ids"],
                "click_markets": user_hist["market_ids"],
                "click_categories": user_hist["category_ids"],
                "query_tokens": query_tokens,
                "candidate_items": candidate_items,
                "label_retriever": label_retriever,
                "label_ctr": label_ctr,
                "label_ctcar": label_ctcar,
                "label_ctcvr": label_ctcvr,
            }

            if is_train:
                train_sessions.append(session)
            else:
                test_sessions.append(session)

        return train_sessions, test_sessions
