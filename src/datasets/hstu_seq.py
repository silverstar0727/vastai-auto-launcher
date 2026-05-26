"""HSTU sequential DataModule.

train interactions parquet 으로부터 user 별 시간순 시퀀스를 구성.
multi-behavior 토큰 포함 (click/like/cart/purchase).

각 학습 sample:
  input_items[T-1], input_behaviors[T-1], position[T-1], pad_mask
  target_item                            (다음 시점의 item)
  target_behavior                        (학습 시 가중치용)

전제:
  meta/active_items.parquet  — item vocab
  interactions/{train,val,test}/*.parquet  — split 이미 적용된 일별 parquet
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader


# behavior id 매핑 — 0=pad
BEHAVIOR_TO_ID: Dict[str, int] = {
    "click": 1, "like": 2, "cart": 3, "purchase": 4,
}


def _load_period(scripts_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "period", scripts_dir / "00_define_period.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------- vocab --------

class _ItemVocab:
    """active_items.parquet 기반 sno → contiguous int id (0=pad)."""

    def __init__(self, active_items_path: Path):
        df = pd.read_parquet(active_items_path, columns=["goods_sno"])
        df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce").astype("int64")
        df = df.dropna().drop_duplicates("goods_sno").reset_index(drop=True)
        self.snos = df["goods_sno"].values
        self.sno_to_id = {int(s): i + 1 for i, s in enumerate(self.snos)}  # 0=pad
        self.num_items = len(self.snos) + 1

    def lookup(self, snos: np.ndarray) -> np.ndarray:
        # vectorized via pandas map (np.searchsorted 도 가능하지만 dict 가 명확)
        out = np.zeros_like(snos, dtype=np.int64)
        for i, s in enumerate(snos):
            out[i] = self.sno_to_id.get(int(s), 0)
        return out


# -------- sequence build --------

def _build_user_long_sequences(
    parquet_files: List[Path],
    vocab: _ItemVocab,
    max_history: int = 1000,   # user 전체 보관할 최대 길이 (augmentation 용)
    min_len: int = 5,
) -> Dict[str, List[np.ndarray]]:
    """user 별 시간순 전체 시퀀스 (jagged) 를 반환. augment 시 sub-sequence sampling.

    Returns:
        users: (N,) str
        items: list of np.int64 array (가변 길이)
        behs:  list of np.int64 array (가변 길이)
    """
    rows = []
    for fp in parquet_files:
        df = pd.read_parquet(fp, columns=["user_code", "goods_sno", "event", "ts"])
        df = df.dropna(subset=["user_code", "goods_sno", "event", "ts"])
        df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce").astype("int64")
        df = df.dropna(subset=["goods_sno"])
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if all_df.empty:
        return {"user_codes": np.array([]), "items": [], "behs": []}

    all_df["item_id"] = vocab.lookup(all_df["goods_sno"].values)
    all_df["beh_id"] = all_df["event"].map(BEHAVIOR_TO_ID).fillna(0).astype(np.int64)
    all_df = all_df[all_df["item_id"] > 0]
    all_df = all_df.sort_values(["user_code", "ts"], kind="stable")

    user_codes: List[str] = []
    items_list: List[np.ndarray] = []
    behs_list: List[np.ndarray] = []
    for uc, g in all_df.groupby("user_code", sort=False):
        items = g["item_id"].values[-max_history:]
        behs = g["beh_id"].values[-max_history:]
        if len(items) < min_len:
            continue
        user_codes.append(uc)
        items_list.append(items)
        behs_list.append(behs)

    return {
        "user_codes": np.array(user_codes),
        "items": items_list,
        "behs": behs_list,
    }


# -------- datasets --------

class _AugmentedTrainDataset(Dataset):
    """학습: sliding sub-sequence augmentation.

    user 전체 시퀀스에서 random end position 을 뽑아
    [end - max_len, end-1] 를 input, end 위치 item 을 target 으로 사용.

    augment_factor=N → user × N samples (data N× 확장).
    user 시퀀스가 짧으면 augment_factor 만큼 못 만들고 가능한 만큼만.
    """

    def __init__(
        self,
        items_list: List[np.ndarray],
        behs_list: List[np.ndarray],
        max_len: int,
        augment_factor: int = 8,
        seed: int = 42,
    ):
        self.items_list = items_list
        self.behs_list = behs_list
        self.max_len = max_len
        self.augment_factor = augment_factor
        rng = np.random.default_rng(seed)

        # 미리 (user_idx, end_pos) pair 생성
        self.samples: List[tuple] = []
        for u_idx, items in enumerate(items_list):
            L = len(items)
            if L < 2:
                continue
            # 가능한 end position: [1, L-1] (target = items[end_pos])
            n = min(augment_factor, L - 1)
            ends = rng.choice(np.arange(1, L), size=n, replace=False)
            for e in ends:
                self.samples.append((u_idx, int(e)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u_idx, end_pos = self.samples[idx]
        items = self.items_list[u_idx]
        behs = self.behs_list[u_idx]
        start = max(0, end_pos - self.max_len)
        in_items = items[start:end_pos]
        in_behs = behs[start:end_pos]
        target_item = int(items[end_pos])
        target_beh = int(behs[end_pos])

        # left-pad to max_len
        item_pad = np.zeros(self.max_len, dtype=np.int64)
        beh_pad = np.zeros(self.max_len, dtype=np.int64)
        L = len(in_items)
        item_pad[-L:] = in_items
        beh_pad[-L:] = in_behs
        positions = np.arange(self.max_len, dtype=np.int64)
        return {
            "item_ids": torch.from_numpy(item_pad),
            "behavior_ids": torch.from_numpy(beh_pad),
            "positions": torch.from_numpy(positions),
            "mask": torch.from_numpy((item_pad > 0).astype(np.int64)).bool(),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_behavior": torch.tensor(target_beh, dtype=torch.long),
        }


class _EvalDataset(Dataset):
    """val/test: 각 user 마지막 1개 item 을 hold-out target."""
    def __init__(self, items_list: List[np.ndarray], behs_list: List[np.ndarray], max_len: int):
        self.items_list = items_list
        self.behs_list = behs_list
        self.max_len = max_len

    def __len__(self):
        return len(self.items_list)

    def __getitem__(self, idx):
        items = self.items_list[idx]
        behs = self.behs_list[idx]
        end_pos = len(items) - 1
        start = max(0, end_pos - self.max_len)
        in_items = items[start:end_pos]
        in_behs = behs[start:end_pos]
        target_item = int(items[end_pos])
        item_pad = np.zeros(self.max_len, dtype=np.int64)
        beh_pad = np.zeros(self.max_len, dtype=np.int64)
        L = len(in_items)
        item_pad[-L:] = in_items
        beh_pad[-L:] = in_behs
        return {
            "item_ids": torch.from_numpy(item_pad),
            "behavior_ids": torch.from_numpy(beh_pad),
            "positions": torch.from_numpy(np.arange(self.max_len, dtype=np.int64)),
            "mask": torch.from_numpy((item_pad > 0).astype(np.int64)).bool(),
            "target_item": torch.tensor(target_item, dtype=torch.long),
        }


# -------- datamodule --------

class HSTUSequentialDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        max_seq_len: int = 200,
        min_seq_len: int = 5,
        max_user_history: int = 1000,
        augment_factor: int = 8,
        batch_size: int = 256,
        eval_batch_size: int = 128,
        num_workers: int = 4,
        train_days: int = 30,
        val_days: int = 7,
        test_days: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.vocab: Optional[_ItemVocab] = None
        self._train_seqs = None
        self._val_seqs = None
        self._test_seqs = None

    @property
    def num_items(self) -> int:
        return self.vocab.num_items if self.vocab else 0

    @property
    def num_behaviors(self) -> int:
        return max(BEHAVIOR_TO_ID.values()) + 1

    def setup(self, stage: Optional[str] = None):
        if self.vocab is not None:
            return
        self.vocab = _ItemVocab(self.base / "meta" / "active_items.parquet")

        train_files = sorted((self.base / "interactions" / "train").glob("*.parquet"))[-self.hparams.train_days:]
        val_files = sorted((self.base / "interactions" / "val").glob("*.parquet"))[-self.hparams.val_days:]
        test_files = sorted((self.base / "interactions" / "test").glob("*.parquet"))[-self.hparams.test_days:]

        self._train_seqs = _build_user_long_sequences(
            train_files, self.vocab,
            max_history=self.hparams.max_user_history, min_len=self.hparams.min_seq_len,
        )
        self._val_seqs = _build_user_long_sequences(
            val_files, self.vocab,
            max_history=self.hparams.max_user_history, min_len=self.hparams.min_seq_len,
        )
        self._test_seqs = _build_user_long_sequences(
            test_files, self.vocab,
            max_history=self.hparams.max_user_history, min_len=self.hparams.min_seq_len,
        )

    def train_dataloader(self):
        ds = _AugmentedTrainDataset(
            self._train_seqs["items"], self._train_seqs["behs"],
            max_len=self.hparams.max_seq_len,
            augment_factor=self.hparams.augment_factor,
            seed=self.hparams.seed,
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        ds = _EvalDataset(self._val_seqs["items"], self._val_seqs["behs"], self.hparams.max_seq_len)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        ds = _EvalDataset(self._test_seqs["items"], self._test_seqs["behs"], self.hparams.max_seq_len)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)
