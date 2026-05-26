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

def _build_user_sequences(
    parquet_files: List[Path],
    vocab: _ItemVocab,
    max_len: int,
    min_len: int = 5,
) -> Dict[str, np.ndarray]:
    """일별 parquet 들을 합쳐 user별 시간순 시퀀스 생성.

    Returns:
        {
          'user_codes': (N,) str array,
          'item_ids':   (N, T) int64 — 0 pad
          'behavior_ids': (N, T) int64 — 0 pad
          'lengths':    (N,)  int64
        }
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
        return {"user_codes": np.array([]), "item_ids": np.zeros((0, max_len), dtype=np.int64),
                "behavior_ids": np.zeros((0, max_len), dtype=np.int64), "lengths": np.zeros(0, dtype=np.int64)}

    all_df["item_id"] = vocab.lookup(all_df["goods_sno"].values)
    all_df["beh_id"] = all_df["event"].map(BEHAVIOR_TO_ID).fillna(0).astype(np.int64)
    all_df = all_df[all_df["item_id"] > 0]
    all_df = all_df.sort_values(["user_code", "ts"], kind="stable")

    user_codes: List[str] = []
    item_seqs: List[np.ndarray] = []
    beh_seqs: List[np.ndarray] = []
    lengths: List[int] = []
    for uc, g in all_df.groupby("user_code", sort=False):
        items = g["item_id"].values
        behs = g["beh_id"].values
        if len(items) < min_len:
            continue
        items = items[-max_len:]
        behs = behs[-max_len:]
        L_ = len(items)
        # left padding
        item_pad = np.zeros(max_len, dtype=np.int64)
        beh_pad = np.zeros(max_len, dtype=np.int64)
        item_pad[-L_:] = items
        beh_pad[-L_:] = behs
        user_codes.append(uc)
        item_seqs.append(item_pad)
        beh_seqs.append(beh_pad)
        lengths.append(L_)

    return {
        "user_codes": np.array(user_codes),
        "item_ids": np.stack(item_seqs) if item_seqs else np.zeros((0, max_len), dtype=np.int64),
        "behavior_ids": np.stack(beh_seqs) if beh_seqs else np.zeros((0, max_len), dtype=np.int64),
        "lengths": np.array(lengths, dtype=np.int64),
    }


# -------- datasets --------

class _NextItemTrainDataset(Dataset):
    """학습: causal next-item prediction. input = seq[:-1], target = seq[1:]."""

    def __init__(self, seqs: Dict[str, np.ndarray], max_len: int):
        self.item = seqs["item_ids"]
        self.beh = seqs["behavior_ids"]
        self.lengths = seqs["lengths"]
        self.max_len = max_len

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        items = self.item[idx]
        behs = self.beh[idx]
        # input: 마지막 1개 제외 / target: 마지막 1개 (next-item)
        # left-pad 형태라 마지막 위치가 시퀀스 끝
        input_items = items.copy()
        input_behs = behs.copy()
        target_item = int(items[-1])
        target_beh = int(behs[-1])
        # input 의 마지막 1개를 pad 로 (causal)
        input_items[-1] = 0
        input_behs[-1] = 0
        mask = (input_items > 0).astype(np.int64)
        positions = np.arange(self.max_len, dtype=np.int64)
        return {
            "item_ids": torch.from_numpy(input_items),
            "behavior_ids": torch.from_numpy(input_behs),
            "positions": torch.from_numpy(positions),
            "mask": torch.from_numpy(mask).bool(),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_behavior": torch.tensor(target_beh, dtype=torch.long),
        }


class _EvalDataset(Dataset):
    """val/test: 같은 형식, target_item 은 hold-out (마지막)."""
    def __init__(self, seqs: Dict[str, np.ndarray], max_len: int):
        self.item = seqs["item_ids"]
        self.beh = seqs["behavior_ids"]
        self.max_len = max_len

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        items = self.item[idx]
        behs = self.beh[idx]
        target_item = int(items[-1])
        input_items = items.copy(); input_items[-1] = 0
        input_behs = behs.copy(); input_behs[-1] = 0
        positions = np.arange(self.max_len, dtype=np.int64)
        return {
            "item_ids": torch.from_numpy(input_items),
            "behavior_ids": torch.from_numpy(input_behs),
            "positions": torch.from_numpy(positions),
            "mask": torch.from_numpy((input_items > 0).astype(np.int64)).bool(),
            "target_item": torch.tensor(target_item, dtype=torch.long),
        }


# -------- datamodule --------

class HSTUSequentialDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        max_seq_len: int = 200,
        min_seq_len: int = 5,
        batch_size: int = 256,
        eval_batch_size: int = 128,
        num_workers: int = 4,
        train_days: int = 14,    # PoC 부담 줄이기 위해 train 마지막 N 일만 우선 사용
        val_days: int = 7,
        test_days: int = 16,
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

        self._train_seqs = _build_user_sequences(train_files, self.vocab,
                                                  self.hparams.max_seq_len, self.hparams.min_seq_len)
        self._val_seqs = _build_user_sequences(val_files, self.vocab,
                                                self.hparams.max_seq_len, self.hparams.min_seq_len)
        self._test_seqs = _build_user_sequences(test_files, self.vocab,
                                                 self.hparams.max_seq_len, self.hparams.min_seq_len)

    def train_dataloader(self):
        ds = _NextItemTrainDataset(self._train_seqs, self.hparams.max_seq_len)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        ds = _EvalDataset(self._val_seqs, self.hparams.max_seq_len)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        ds = _EvalDataset(self._test_seqs, self.hparams.max_seq_len)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)
