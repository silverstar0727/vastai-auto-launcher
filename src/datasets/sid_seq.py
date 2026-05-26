"""TIGER-lite sequential DataModule.

전제:
  - meta/active_items.parquet
  - meta/item_to_sid.parquet  (SID Tokenizer 학습 산출물)
    컬럼: goods_sno, sid_0, sid_1, sid_2 (각 레벨 codebook index, 0-based)
  - interactions/{train,val,test}/*.parquet

토큰화:
  - 시퀀스의 각 item 을 SID 토큰 L 개로 flatten → encoder 입력 (T_enc = N × L)
  - target: 마지막 item 의 SID (decoder)
  - behavior_id 는 item 별로 broadcast (L 번 반복)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader

from nets.tiger_lite import build_sid_vocab

BEHAVIOR_TO_ID = {"click": 1, "like": 2, "cart": 3, "purchase": 4}
PAD_ID, BOS_ID, SEP_ID, EOS_ID = 0, 1, 2, 3


def _absolute_sid(level_codes: np.ndarray, offsets: List[int]) -> np.ndarray:
    """(N, L) per-level codes → (N, L) absolute token ids."""
    off = np.array(offsets, dtype=np.int64)[None, :]
    return level_codes + off


class _SIDStore:
    """goods_sno → absolute SID tuple (length L)."""

    def __init__(self, item_to_sid_df: pd.DataFrame, codebook_sizes: Tuple[int, ...]):
        offsets, total_vocab = build_sid_vocab(codebook_sizes)
        self.offsets = offsets
        self.total_vocab = total_vocab
        self.L = len(codebook_sizes)
        # 검증: per-level codes 가 [0, codebook_size) 안에 있는지
        codes = item_to_sid_df[[f"sid_{l}" for l in range(self.L)]].values.astype(np.int64)
        for l in range(self.L):
            assert codes[:, l].max() < codebook_sizes[l], f"sid_{l} 가 codebook 범위 밖"
        abs_codes = _absolute_sid(codes, offsets)
        self.sno_to_sid: Dict[int, np.ndarray] = {
            int(sno): abs_codes[i] for i, sno in enumerate(item_to_sid_df["goods_sno"].astype(int))
        }

    def lookup(self, snos: List[int]) -> Optional[np.ndarray]:
        out = []
        for s in snos:
            sid = self.sno_to_sid.get(int(s))
            if sid is None:
                return None
            out.append(sid)
        return np.stack(out) if out else None

    def all_item_sids(self) -> torch.Tensor:
        arr = np.stack(list(self.sno_to_sid.values()))
        return torch.from_numpy(arr)


class _SIDSeqTrainDataset(Dataset):
    """학습: encoder = user history items (SID flat), decoder = next item SID."""

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "enc_tokens": torch.from_numpy(s["enc_tokens"]),
            "enc_beh": torch.from_numpy(s["enc_beh"]),
            "enc_pos": torch.from_numpy(s["enc_pos"]),
            "enc_mask": torch.from_numpy(s["enc_mask"]).bool(),
            "dec_input": torch.from_numpy(s["dec_input"]),
            "dec_target": torch.from_numpy(s["dec_target"]),
            "dec_pos": torch.from_numpy(s["dec_pos"]),
            "target_behavior": torch.tensor(s["target_behavior"], dtype=torch.long),
        }


class TIGERLiteDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        codebook_sizes: tuple = (2048, 1024, 512),
        max_history_items: int = 50,         # T_enc = 50 × 3 = 150
        min_history_items: int = 5,
        batch_size: int = 128,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        train_days: int = 14,
        val_days: int = 7,
        test_days: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.store: Optional[_SIDStore] = None
        self._train_samples = None
        self._val_samples = None

    @property
    def num_behaviors(self):
        return max(BEHAVIOR_TO_ID.values()) + 1

    @property
    def total_vocab(self):
        return self.store.total_vocab if self.store else 0

    def setup(self, stage: Optional[str] = None):
        if self.store is not None:
            return
        item_sid = pd.read_parquet(self.base / "meta" / "item_to_sid.parquet")
        item_sid["goods_sno"] = pd.to_numeric(item_sid["goods_sno"], errors="coerce").astype("int64")
        self.store = _SIDStore(item_sid, tuple(self.hparams.codebook_sizes))

        train_files = sorted((self.base / "interactions" / "train").glob("*.parquet"))[-self.hparams.train_days:]
        val_files = sorted((self.base / "interactions" / "val").glob("*.parquet"))[-self.hparams.val_days:]
        self._train_samples = self._build_samples(train_files)
        self._val_samples = self._build_samples(val_files)

    def _build_samples(self, files: List[Path]) -> List[Dict[str, np.ndarray]]:
        rows = []
        for fp in files:
            df = pd.read_parquet(fp, columns=["user_code", "goods_sno", "event", "ts"])
            rows.append(df)
        if not rows:
            return []
        all_df = pd.concat(rows, ignore_index=True).dropna()
        all_df["goods_sno"] = pd.to_numeric(all_df["goods_sno"], errors="coerce").astype("int64")
        all_df = all_df.dropna(subset=["goods_sno"]).sort_values(["user_code", "ts"], kind="stable")

        L = self.store.L
        max_T_enc = self.hparams.max_history_items * L
        samples: List[Dict[str, np.ndarray]] = []

        for uc, g in all_df.groupby("user_code", sort=False):
            snos = g["goods_sno"].tolist()
            evs = g["event"].tolist()
            if len(snos) < self.hparams.min_history_items + 1:
                continue
            target_sno = snos[-1]
            target_event = evs[-1]
            history_snos = snos[:-1][-self.hparams.max_history_items:]
            history_evs = evs[:-1][-self.hparams.max_history_items:]

            target_sid = self.store.lookup([target_sno])
            history_sids = self.store.lookup(history_snos)
            if target_sid is None or history_sids is None:
                continue
            target_sid = target_sid[0]
            n_hist = history_sids.shape[0]

            # encoder tokens (T_enc 길이, left-padded)
            enc_tokens = np.zeros(max_T_enc, dtype=np.int64)
            enc_beh = np.zeros(max_T_enc, dtype=np.int64)
            flat = history_sids.flatten()
            flat_beh = np.repeat(
                [BEHAVIOR_TO_ID.get(e, 0) for e in history_evs], L
            ).astype(np.int64)
            enc_tokens[-n_hist * L:] = flat
            enc_beh[-n_hist * L:] = flat_beh

            enc_pos = np.arange(max_T_enc, dtype=np.int64)
            enc_mask = (enc_tokens > 0).astype(np.int64)

            # decoder input = [BOS, sid_0, sid_1, ..., sid_{L-1}]
            dec_input = np.concatenate([[BOS_ID], target_sid]).astype(np.int64)
            # target = [sid_0, sid_1, ..., sid_{L-1}, EOS]
            dec_target = np.concatenate([target_sid, [EOS_ID]]).astype(np.int64)
            dec_pos = np.arange(dec_input.shape[0], dtype=np.int64)

            samples.append({
                "enc_tokens": enc_tokens,
                "enc_beh": enc_beh,
                "enc_pos": enc_pos,
                "enc_mask": enc_mask,
                "dec_input": dec_input,
                "dec_target": dec_target,
                "dec_pos": dec_pos,
                "target_behavior": BEHAVIOR_TO_ID.get(target_event, 0),
            })
        return samples

    def train_dataloader(self):
        ds = _SIDSeqTrainDataset(self._train_samples)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        ds = _SIDSeqTrainDataset(self._val_samples)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)
