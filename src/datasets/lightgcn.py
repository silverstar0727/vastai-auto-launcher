"""LightGCN DataModule.

기존 lightgcn/train/factory.py + dataloaders.py를
Lightning DataModule 패턴으로 통합.

CBF/Complement DataModule과의 핵심 차이점:
1. Parquet 파일 기반 이벤트 데이터 (CSV 인터랙션이 아님)
2. 자체 label encoder (user + item) 사용
3. Graph 구축을 위한 전처리 (degree_inv_sqrt, item_freq)
4. 유저 시퀀스 구축 (seq_items, seq_offsets)
5. Cold-start dropout 지원
"""

import glob
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torch import Tensor

import lightning as L

from nets.lightgcn.graph_loader import GraphLoader
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix

logger = logging.getLogger(__name__)


# --- Dataset ---


class LightgcnHistoryDataset(data_utils.Dataset):
    """History Fusion Gate 학습용 Dataset.

    유저 히스토리를 flat array(seq_items) + offset 배열(seq_offsets)로 관리한다.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        seq_items: np.ndarray,
        seq_offsets: np.ndarray,
        n_items: int,
        num_neg_samples: int,
        max_history_len: int,
        cold_start_dropout: float,
    ):
        self.users = events["user_id"].values
        self.items = events["item_id"].values
        self.seq_lens = events["seq_len"].values
        self.seq_items = seq_items
        self.seq_offsets = seq_offsets
        self.n_items = n_items
        self.num_neg_samples = num_neg_samples
        self.max_history_len = max_history_len
        self.cold_start_dropout = cold_start_dropout

    def __len__(self) -> int:
        return len(self.users)

    def _get_history(self, user_id: int, idx: int) -> tuple[np.ndarray, int]:
        offset = self.seq_offsets[user_id]
        seq_len = int(self.seq_lens[idx])
        hist_start = max(0, seq_len - self.max_history_len)
        hist_len = seq_len - hist_start
        history = self.seq_items[offset + hist_start : offset + seq_len]
        return history, hist_len

    def __getitem__(self, idx: int) -> dict:
        user_id = int(self.users[idx])
        item_id = int(self.items[idx])
        neg_item_ids = np.random.randint(1, self.n_items + 1, self.num_neg_samples)

        history, hist_len = self._get_history(user_id, idx)

        # cold-start dropout
        if hist_len > 0 and self.cold_start_dropout > 0 and np.random.random() < self.cold_start_dropout:
            user_id = 0

        history_item_ids = np.zeros(self.max_history_len, dtype=np.int64)
        history_mask = np.zeros(self.max_history_len, dtype=np.bool_)
        if hist_len > 0:
            history_item_ids[:hist_len] = history
            history_mask[:hist_len] = True

        return {
            "user_id": user_id,
            "item_id": item_id,
            "neg_item_ids": neg_item_ids,
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "history_len": hist_len,
        }


# --- DataModule ---


class LightGCNDataModule(L.LightningDataModule):
    def __init__(
        self,
        # 데이터 경로
        raw_dataset_root: str = "",
        model_path: str = "",
        event_data_dir: str = "events",
        # 데이터 필터링
        max_users: int = 2_000_000,
        max_items: int = 1_000_000,
        valid_ratio: float = 0.01,
        # 배치 설정
        batch_size: int = 8192,
        test_batch_size: int = 8192,
        num_workers: int = 8,
        # Negative sampling
        num_neg_samples: int = 0,
        # History
        max_history_len: int = 50,
        cold_start_dropout: float = 0.1,
        use_learned_gate: bool = True,
        # Log-Q correction
        logq_correction: bool = True,
        loss_type: str = "ce",
        use_inbatch_neg: bool = True,
        # 캐싱
        save_preprocessed_data: bool = False,
        load_preprocessed_data: bool = False,
        # 시드
        dataloader_random_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        # setup()에서 채워질 속성 (Model에서 읽어감)
        self.num_users: Optional[int] = None
        self.num_items: Optional[int] = None
        self.train_graph_loader: Optional[GraphLoader] = None
        self.degree_inv_sqrt: Optional[Tensor] = None
        self.item_freq: Optional[Tensor] = None

        self._train_dataset = None
        self._eval_dataset = None
        self._train_events = None
        self._valid_events = None

    def setup(self, stage=None):
        if self._train_dataset is not None:
            return

        hp = self.hparams
        cache_dir = os.path.join(hp.raw_dataset_root, "input/data/train_data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(hp.model_path, exist_ok=True)

        # 1. 전처리 또는 캐시 로드
        if hp.load_preprocessed_data:
            data = self._load_cache(cache_dir)
        else:
            data = None

        if data is None:
            data = self._preprocess(cache_dir)

        self.num_users = data["n_users"]
        self.num_items = data["n_items"]
        self._train_events = data["train_events"]
        self._valid_events = data["valid_events"]
        seq_items = data["seq_items"]
        seq_offsets = data["seq_offsets"]
        degree_inv_sqrt_np = data["degree_inv_sqrt"]

        self.degree_inv_sqrt = torch.from_numpy(degree_inv_sqrt_np)

        logger.info(f"User cnt: {self.num_users:,}, Item cnt: {self.num_items:,}")
        logger.info(f"Train cnt: {len(self._train_events):,}, Valid cnt: {len(self._valid_events):,}")

        # 2. Graph loader (train events로만 구축)
        self.train_graph_loader = GraphLoader(self._train_events, self.num_users, self.num_items)
        logger.info(
            f"Graph built: Node cnt={self.train_graph_loader.n_nodes:,}, "
            f"Edge cnt={self.train_graph_loader.n_edges:,}"
        )

        # 3. Item frequency (log-Q correction용)
        use_logq = hp.logq_correction and hp.loss_type == "ce" and hp.use_inbatch_neg
        if use_logq:
            self.item_freq = self._compute_item_freq(self._train_events, self.num_items)

        # 4. Dataset 생성
        cold_start = hp.cold_start_dropout if hp.use_learned_gate else 0.0
        self._train_dataset = LightgcnHistoryDataset(
            self._train_events, seq_items, seq_offsets,
            self.num_items, hp.num_neg_samples, hp.max_history_len, cold_start,
        )
        self._eval_dataset = LightgcnHistoryDataset(
            self._valid_events, seq_items, seq_offsets,
            self.num_items, hp.num_neg_samples, hp.max_history_len, cold_start_dropout=0.0,
        )

    def train_dataloader(self):
        persistent = self.hparams.num_workers > 0
        return data_utils.DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent,
        )

    def val_dataloader(self):
        persistent = self.hparams.num_workers > 0
        return data_utils.DataLoader(
            self._eval_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def create_export_graph_loader(self) -> GraphLoader:
        """학습 후 전체 이벤트(train+valid)로 그래프를 구축하여 반환."""
        all_events = pd.concat([self._train_events, self._valid_events], ignore_index=True)
        return GraphLoader(all_events, self.num_users, self.num_items)

    # --- Private methods ---

    def _preprocess(self, cache_dir: str) -> dict:
        hp = self.hparams

        events = self._load_events()
        events = self._filter_by_col(events, "item_id", hp.max_items)
        events = self._filter_by_col(events, "user_id", hp.max_users)

        n_items = events["item_id"].nunique()
        n_users = events["user_id"].nunique()

        events, user_encoder, item_encoder = self._map_id2index(events)
        train_events, valid_events = self._split_train_valid(events)

        seq_items, seq_offsets = self._build_user_sequences(train_events, n_users)
        train_events = self._add_seq_len(train_events, seq_offsets, is_train=True)
        valid_events = self._add_seq_len(valid_events, seq_offsets, is_train=False)

        degree_inv_sqrt = self._compute_degree_inv_sqrt(train_events, n_items)

        # Save label encoders and metadata
        user_encoder.save(cache_dir, LabelEncoderPrefix.USER)
        item_encoder.save(cache_dir, LabelEncoderPrefix.ITEM)
        with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
            json.dump({"n_users": n_users, "n_items": n_items}, f)
        np.save(os.path.join(cache_dir, "degree_inv_sqrt.npy"), degree_inv_sqrt)

        if hp.save_preprocessed_data:
            train_events.to_parquet(os.path.join(cache_dir, "train_events.parquet"), index=False)
            valid_events.to_parquet(os.path.join(cache_dir, "valid_events.parquet"), index=False)
            np.save(os.path.join(cache_dir, "seq_items.npy"), seq_items)
            np.save(os.path.join(cache_dir, "seq_offsets.npy"), seq_offsets)
            logger.info(f"Preprocessing cache saved to {cache_dir}")

        return {
            "n_users": n_users, "n_items": n_items,
            "train_events": train_events, "valid_events": valid_events,
            "seq_items": seq_items, "seq_offsets": seq_offsets,
            "degree_inv_sqrt": degree_inv_sqrt,
        }

    def _load_events(self) -> pd.DataFrame:
        hp = self.hparams
        parquet_dir = os.path.join(hp.raw_dataset_root, "input/data", hp.event_data_dir)
        parquet_files = glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
        events = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        events.drop_duplicates(inplace=True)
        return events.rename(columns={"user_code": "user_id", "goods_sno": "item_id"})

    @staticmethod
    def _filter_by_col(events: pd.DataFrame, col: str, max_keys: int) -> pd.DataFrame:
        key_cnt = events[col].value_counts(sort=True)
        n_keys_before = len(key_cnt)
        if n_keys_before <= max_keys:
            logger.info(f"Filter {col}: skip ({n_keys_before} <= {max_keys})")
            return events
        keys_to_drop = key_cnt[max_keys:].index.values
        events = events[~events[col].isin(keys_to_drop)]
        logger.info(f"Filter {col}: {n_keys_before} -> {n_keys_before - len(keys_to_drop)}")
        return events

    @staticmethod
    def _map_id2index(events: pd.DataFrame):
        user_encoder = LabelEncoder.from_item_ids(events["user_id"], id_dtype=str)
        item_encoder = LabelEncoder.from_item_ids(events["item_id"], id_dtype=str)
        events["user_id"] = events["user_id"].astype(str).map(user_encoder.item_id2index)
        events["item_id"] = events["item_id"].astype(str).map(item_encoder.item_id2index)
        events = events.dropna(subset=["user_id", "item_id"])
        return events, user_encoder, item_encoder

    def _split_train_valid(self, events: pd.DataFrame):
        hp = self.hparams
        random.seed(hp.dataloader_random_seed)
        valid_indices = []
        for _, group in events.groupby("user_id"):
            if len(group) < 2:
                continue
            n_valid = max(1, int(len(group) * hp.valid_ratio))
            n_valid = min(n_valid, len(group) - 1)
            valid_indices.extend(random.sample(group.index.tolist(), k=n_valid))
        valid_mask = events.index.isin(valid_indices)
        train_events = events[~valid_mask].reset_index(drop=True)
        valid_events = events[valid_mask].reset_index(drop=True)
        return train_events, valid_events

    @staticmethod
    def _build_user_sequences(train_events: pd.DataFrame, n_users: int):
        sorted_events = train_events.sort_values(["user_id", "synced_occurred_ts"])
        seq_items = sorted_events["item_id"].values.astype(np.int64)
        sizes = sorted_events["user_id"].value_counts(sort=False).sort_index().values
        seq_offsets = np.zeros(n_users + 2, dtype=np.int64)
        seq_offsets[2:] = np.cumsum(sizes)
        return seq_items, seq_offsets

    @staticmethod
    def _add_seq_len(events: pd.DataFrame, seq_offsets: np.ndarray, is_train: bool):
        if is_train:
            events = events.sort_values(["user_id", "synced_occurred_ts"]).reset_index(drop=True)
            events["seq_len"] = events.groupby("user_id").cumcount().values
        else:
            user_ids = events["user_id"].astype(int).values
            events["seq_len"] = seq_offsets[user_ids + 1] - seq_offsets[user_ids]
        return events

    @staticmethod
    def _compute_degree_inv_sqrt(train_events: pd.DataFrame, n_items: int) -> np.ndarray:
        counts = train_events["item_id"].value_counts()
        inv_sqrt = np.zeros(n_items + 1, dtype=np.float32)
        inv_sqrt[counts.index.values] = np.power(counts.values.astype(np.float32), -0.5)
        return inv_sqrt

    @staticmethod
    def _compute_item_freq(train_events: pd.DataFrame, n_items: int) -> Tensor:
        freq = torch.zeros(n_items + 1)
        counts = train_events["item_id"].value_counts()
        freq[counts.index.values] = torch.from_numpy(counts.values.astype(np.float32))
        freq = freq / freq.sum()
        return freq

    def _load_cache(self, cache_dir: str):
        metadata_path = os.path.join(cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
        logger.info(f"Loading preprocessing cache from {cache_dir}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        return {
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
            "train_events": pd.read_parquet(os.path.join(cache_dir, "train_events.parquet")),
            "valid_events": pd.read_parquet(os.path.join(cache_dir, "valid_events.parquet")),
            "seq_items": np.load(os.path.join(cache_dir, "seq_items.npy")),
            "seq_offsets": np.load(os.path.join(cache_dir, "seq_offsets.npy")),
            "degree_inv_sqrt": np.load(os.path.join(cache_dir, "degree_inv_sqrt.npy")),
        }
