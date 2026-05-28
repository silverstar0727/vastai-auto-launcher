"""TIGER-Gemma CPT (Continued Pre-Training) DataModule.

PLUM CPT 패턴 (50:50 mix):
  1) User behavior 시퀀스 — next-token prediction:
     "<BEH_click> <SID_0_521><SID_1_187><SID_2_92> <BEH_cart> <SID_0_..> ..."
  2) Item metadata 텍스트 — next-token prediction:
     "<SID_0_521><SID_1_187><SID_2_92> brand_sno 1234 category 원피스 price 25000"

학습 목표: Gemma 가 SID 토큰과 자연어/카테고리 사이의 관계, 그리고 user 행동 시퀀스
의 next-item 패턴을 함께 학습.

전제:
  - meta/item_to_sid.parquet (sid_0/1/2 per goods_sno)
  - meta/item_meta.parquet (name, brand_sno, category 등)
  - interactions/train/{date}.parquet
  - GemmaTigerBackbone.tokenizer (vocab 확장 완료)
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

BEHAVIOR_TO_NAME = {"click": "<BEH_click>", "like": "<BEH_like>",
                     "cart": "<BEH_cart>", "purchase": "<BEH_purchase>"}


def _load_period(scripts_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "period", scripts_dir / "00_define_period.py"
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class _CPTDataset(Dataset):
    """User-seq + item-meta mixed text-LM dataset (이미 토큰화 된 ids).

    samples: list of {"input_ids": np.array, "attention_mask": np.array, "labels": np.array}
    """
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.from_numpy(s["input_ids"]),
            "attention_mask": torch.from_numpy(s["attention_mask"]),
            "labels": torch.from_numpy(s["labels"]),
        }


class TIGERGemmaCPTDataModule(L.LightningDataModule):
    """CPT 데이터 — text LM (user seq + item meta).

    Args:
        base_dir: PoC 데이터 루트
        max_seq_len: tokenized 시퀀스 최대 길이 (예: 1024)
        max_history_items: user 시퀀스에서 보존할 최근 N item
        max_train_user_samples / max_train_item_samples: 학습 sample 수 cap
        ratio_user_to_item: user-seq 비율 (0.5 = PLUM 50:50)
        train_days: 사용할 train 일수
    """

    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        model_name: str = "google/gemma-4-E4B",     # tokenizer 로드용
        codebook_sizes: tuple = (2048, 1024, 512),   # SID 토큰 확장용
        max_seq_len: int = 1024,
        max_history_items: int = 50,
        max_train_user_samples: int = 1_000_000,
        max_train_item_samples: int = 1_000_000,
        ratio_user_to_item: float = 0.5,
        train_days: int = 14,
        val_days: int = 7,
        batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.tokenizer = None  # setup() 에서 자체 로드 (model 과 같은 vocab 확장)
        self._train_samples: Optional[List[Dict[str, np.ndarray]]] = None
        self._val_samples: Optional[List[Dict[str, np.ndarray]]] = None

    def set_tokenizer(self, tokenizer):
        """선택: 외부에서 tokenizer 강제 주입 (model.backbone.tokenizer 와 공유 보장)."""
        self.tokenizer = tokenizer

    def _init_tokenizer(self):
        """model 과 같은 vocab 확장 (deterministic)."""
        from transformers import AutoTokenizer
        from nets.tiger_lite.gemma_backbone import (
            make_sid_tokens, BEHAVIOR_TOKENS, STRUCTURAL_TOKENS,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        new_tokens = (
            make_sid_tokens(tuple(self.hparams.codebook_sizes))
            + BEHAVIOR_TOKENS + STRUCTURAL_TOKENS
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    def setup(self, stage: Optional[str] = None):
        if self._train_samples is not None:
            return
        if self.tokenizer is None:
            self._init_tokenizer()

        # SID 매핑 로드
        sid_df = pd.read_parquet(self.base / "meta" / "item_to_sid.parquet")
        sid_df["goods_sno"] = sid_df["goods_sno"].astype("int64")
        self.sno_to_sid = {
            int(r.goods_sno): (int(r.sid_0), int(r.sid_1), int(r.sid_2))
            for r in sid_df.itertuples(index=False)
        }

        # item meta 로드
        meta_df = pd.read_parquet(
            self.base / "meta" / "item_meta.parquet",
            columns=["goods_sno", "name", "brand_sno", "price",
                     "category_sno", "standard_category_sno"],
        )
        meta_df["goods_sno"] = meta_df["goods_sno"].astype("int64")
        self.meta = {int(r.goods_sno): r._asdict() for r in meta_df.itertuples(index=False)}

        # train interactions
        train_files = sorted((self.base / "interactions" / "train").glob("*.parquet"))[-self.hparams.train_days:]
        val_files = sorted((self.base / "interactions" / "val").glob("*.parquet"))[-self.hparams.val_days:]
        self._train_samples = self._build_samples(train_files, split="train")
        self._val_samples = self._build_samples(val_files, split="val")

    def _build_samples(self, files, split: str) -> List[Dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.hparams.seed if split == "train" else 0)
        samples = []

        # === part 1: user-seq text ===
        n_user_target = self.hparams.max_train_user_samples if split == "train" else max(2000, self.hparams.max_train_user_samples // 10)
        user_samples = self._build_user_seq_samples(files, rng, n_user_target)
        samples.extend(user_samples)

        # === part 2: item-meta text ===
        n_item_target = self.hparams.max_train_item_samples if split == "train" else max(2000, self.hparams.max_train_item_samples // 10)
        item_samples = self._build_item_meta_samples(rng, n_item_target)
        samples.extend(item_samples)

        # shuffle
        rng.shuffle(samples)
        return samples

    def _build_user_seq_samples(self, files, rng, n_target: int) -> List[Dict[str, np.ndarray]]:
        out = []
        for fp in files:
            if len(out) >= n_target:
                break
            df = pd.read_parquet(fp, columns=["user_code", "goods_sno", "event", "ts"])
            df = df.dropna()
            df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce").astype("int64")
            df = df.dropna(subset=["goods_sno"]).sort_values(["user_code", "ts"], kind="stable")

            for uc, g in df.groupby("user_code", sort=False):
                if len(out) >= n_target:
                    break
                snos = g["goods_sno"].tolist()[-self.hparams.max_history_items:]
                evs = g["event"].tolist()[-self.hparams.max_history_items:]
                if len(snos) < 5:
                    continue
                # 텍스트화
                parts = []
                for s, e in zip(snos, evs):
                    sid = self.sno_to_sid.get(int(s))
                    if sid is None:
                        continue
                    beh_tok = BEHAVIOR_TO_NAME.get(e, "<BEH_click>")
                    parts.append(beh_tok)
                    parts.append(f"<SID_0_{sid[0]}>")
                    parts.append(f"<SID_1_{sid[1]}>")
                    parts.append(f"<SID_2_{sid[2]}>")
                    parts.append("<SEP_ITEM>")
                if len(parts) < 8:
                    continue
                text = "".join(parts)
                ids = self._tokenize_and_pad(text)
                if ids is not None:
                    out.append(ids)
        return out

    def _build_item_meta_samples(self, rng, n_target: int) -> List[Dict[str, np.ndarray]]:
        out = []
        snos_pool = list(self.sno_to_sid.keys())
        if len(snos_pool) > n_target:
            idx = rng.choice(len(snos_pool), n_target, replace=False)
            snos_pool = [snos_pool[i] for i in idx]
        for sno in snos_pool:
            sid = self.sno_to_sid[sno]
            m = self.meta.get(int(sno))
            if m is None:
                continue
            # text format: SID + meta (NaN-safe)
            def _to_int(x):
                try:
                    if x is None or pd.isna(x):
                        return 0
                    return int(x)
                except (TypeError, ValueError):
                    return 0
            text = (
                f"<SID_0_{sid[0]}><SID_1_{sid[1]}><SID_2_{sid[2]}>"
                f" name: {str(m.get('name', '') or '')}"
                f" brand: {_to_int(m.get('brand_sno'))}"
                f" category: {_to_int(m.get('standard_category_sno'))}"
                f" price: {_to_int(m.get('price'))}"
            )
            ids = self._tokenize_and_pad(text)
            if ids is not None:
                out.append(ids)
        return out

    def _tokenize_and_pad(self, text: str) -> Optional[Dict[str, np.ndarray]]:
        enc = self.tokenizer(
            text,
            max_length=self.hparams.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        labels = ids.copy()
        labels[mask == 0] = -100  # pad → ignore in CE
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def train_dataloader(self):
        ds = _CPTDataset(self._train_samples)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        ds = _CPTDataset(self._val_samples)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)
