"""Multi-Interest DataModule for v2_full PoC data.

Production multi_interest 의 CSV+LabelEncoder 전처리 파이프라인을 우회하고,
v2_full 의 parquet 를 직접 읽어 동일 텐서 포맷으로 제공한다. 모델·네트워크
코드 (src/models/multi_interest.py, src/nets/multi_interest/...) 는 그대로 사용.

설계 doc: data/reco_experiments/v2_full/EXPERIMENT_MULTI_INTEREST_VS_TIGER.md

Production 정합 (`docs/inhouse_kube/src/apps/multi_interest/pipeline_config.py`):
  - max_items 300,000 (popular top-K)
  - min_actions_per_user 100
  - preference_event_weight 40.0, next_items_window_size 5
  - num_train_sample_per_user 60
  - 데이터 윈도우: production 은 1 년치 (`prod_1yr_train_data_v2`).
    v2_full 에는 train 영역에 98 일만 있어 가용 최대(=98일) 사용.

데이터 소스 (v2_full):
  meta/item_meta.parquet            goods_sno + standard_category_sno
  interactions/{train,val,test}/*.parquet   user_code, goods_sno, event, ts
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch.utils.data as data_utils

import lightning as L

from datasets.multi_interest import (
    MultiInterestEvalDataset,
    MultiInterestTrainDataset,
)
from utils.constants import (
    EVENT_MAP,
    EVENT_VOCA,
    DatasetField,
    FeatureField,
)

logger = logging.getLogger(__name__)


class _V2DatasetHandler:
    """utils.data_handler.DatasetHandler 의 최소 인터페이스 stand-in.

    MultiInterestTrainDataset / MultiInterestEvalDataset 가 호출하는 메서드만
    구현한다:
      - num_train_samples() → user 수
      - num_test_samples()  → eval 대상 sample 수
      - get_user2items(idx, "train") → DataFrame[ITEM_INDEX, EVENT_CODE]
      - get_items(idx, "test")       → list of item_index (next-item positives)
    """

    def __init__(
        self,
        user_train_items: List[np.ndarray],
        user_train_events: List[np.ndarray],
        user_test_items: List[np.ndarray],
    ):
        self.user_train_items = user_train_items
        self.user_train_events = user_train_events
        self.user_test_items = user_test_items

    def num_train_samples(self) -> int:
        return len(self.user_train_items)

    def num_test_samples(self) -> int:
        return len(self.user_test_items)

    def get_user2items(self, user_index: int, split: str) -> pd.DataFrame:
        items = self.user_train_items[user_index]
        events = self.user_train_events[user_index]
        return pd.DataFrame(
            {DatasetField.ITEM_INDEX: items, DatasetField.EVENT_CODE: events}
        )

    def get_items(self, user_index: int, split: str) -> list:
        if split == "test":
            return list(self.user_test_items[user_index])
        raise ValueError(f"unknown split: {split}")


class _NonEmptyEvalSubset(data_utils.Dataset):
    """positives 가 비어있지 않은 user 만 노출하는 wrapper.

    v2_full 의 val/test 기간에 해당 user 의 interaction 이 없거나 선택된 item
    universe 바깥이면 positive 가 비어 평가 불가능 → 그 user 는 제외.
    """

    def __init__(self, base_eval: MultiInterestEvalDataset, positives: List[np.ndarray]):
        self.base = base_eval
        self.indices = [i for i, p in enumerate(positives) if len(p) > 0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


class MultiInterestV2DataModule(L.LightningDataModule):
    """v2_full PoC 데이터로 Multi-Interest 모델 학습/평가."""

    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        # 데이터 윈도우
        train_days: int = 98,
        val_days: int = 7,
        test_days: int = 16,
        # production 필터
        min_actions_per_user: int = 100,
        max_items: int = 300_000,
        # 시퀀스
        max_len: int = 80,
        # 배치
        batch_size: int = 2048,
        test_batch_size: int = 64,
        num_workers: int = 2,
        drop_last_batch: bool = True,
        # 학습 샘플
        num_train_sample_per_user: int = 60,
        preference_event_weight: float = 40.0,
        next_items_window_size: int = 5,
        final_item_sampling_weight: float = 3.0,
        final_item_score: float = 1.0,
        use_rank_learning: bool = False,
        # val cap (TIGER-lite v6 와 정합: 1920 user)
        max_val_samples: Optional[int] = 1920,
        # 시드
        dataloader_random_seed: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        # MultiInterestModel.setup() 이 읽어가는 속성
        self.num_items: Optional[int] = None
        self.num_standard_categories: Optional[int] = None
        self.item_feat_values: Optional[Dict[str, np.ndarray]] = None
        self.item_train_weights = None
        self.goods_info = None

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._sno_to_idx: Optional[Dict[int, int]] = None

    # ------------- public attrs (model setup() 호환) -------------

    @property
    def sno_to_idx(self) -> Dict[int, int]:
        return self._sno_to_idx

    # ------------- setup -------------

    def setup(self, stage=None):
        if self._train_dataset is not None:
            return

        hp = self.hparams
        base = Path(hp.base_dir)

        # 1) train interactions
        train_files = sorted((base / "interactions" / "train").glob("*.parquet"))[-hp.train_days:]
        logger.info(f"[MI-v2] train files: {len(train_files)} (last {hp.train_days} days)")
        train_df = self._load_interactions(train_files)
        logger.info(f"[MI-v2] train rows raw: {len(train_df):,}")

        # 2) item universe — popular top-N from train
        item_freq = train_df["goods_sno"].value_counts()
        top_items = item_freq.head(hp.max_items).index.to_numpy()
        item_set = set(int(s) for s in top_items)
        sorted_top = np.sort(top_items)
        self._sno_to_idx = {int(s): i + 1 for i, s in enumerate(sorted_top)}  # 0 = PAD
        self.num_items = len(sorted_top) + 1
        logger.info(f"[MI-v2] item universe: {len(sorted_top):,} (PAD=0, idx=1..{self.num_items-1})")

        # 3) item → standard_category 매핑
        self._build_category_mapping(base / "meta" / "item_meta.parquet")

        # 4) train data → per-user arrays
        train_df = train_df[train_df["goods_sno"].isin(item_set)].copy()
        train_df["event_code"] = (
            train_df["event"].map(EVENT_MAP).map(EVENT_VOCA).fillna(0).astype("int64")
        )
        train_df["item_idx"] = train_df["goods_sno"].map(self._sno_to_idx).astype("int64")
        train_df = train_df.sort_values(["user_code", "ts"], kind="stable")

        user_counts = train_df.groupby("user_code").size()
        eligible_users = user_counts[user_counts >= hp.min_actions_per_user].index
        train_df = train_df[train_df["user_code"].isin(eligible_users)]
        logger.info(
            f"[MI-v2] eligible users (≥{hp.min_actions_per_user} actions): {len(eligible_users):,}, "
            f"filtered rows: {len(train_df):,}"
        )

        user_train_items: List[np.ndarray] = []
        user_train_events: List[np.ndarray] = []
        eligible_user_codes: List[int] = []
        _ui = 0
        for uc, g in train_df.groupby("user_code", sort=False):
            user_train_items.append(g["item_idx"].to_numpy(dtype=np.int64))
            user_train_events.append(g["event_code"].to_numpy(dtype=np.int64))
            eligible_user_codes.append(int(uc))
            _ui += 1
            if _ui % 200_000 == 0:
                logger.info(f"[MI-v2]   per-user array build {_ui:,}/{len(eligible_users):,}")
        user_to_idx = {uc: i for i, uc in enumerate(eligible_user_codes)}

        # 5) val / test positives (per user 첫 next-item)
        val_positives = self._build_eval_positives(
            base / "interactions" / "val",
            days=hp.val_days,
            item_set=item_set,
            user_to_idx=user_to_idx,
            num_users=len(eligible_user_codes),
            max_users=hp.max_val_samples,
            split_label="val",
        )
        test_positives = self._build_eval_positives(
            base / "interactions" / "test",
            days=hp.test_days,
            item_set=item_set,
            user_to_idx=user_to_idx,
            num_users=len(eligible_user_codes),
            max_users=None,
            split_label="test",
        )

        # 6) Dataset 생성 (production code 재사용)
        val_handler = _V2DatasetHandler(user_train_items, user_train_events, val_positives)
        test_handler = _V2DatasetHandler(user_train_items, user_train_events, test_positives)
        rng = random.Random(int(hp.dataloader_random_seed))

        self._train_dataset = MultiInterestTrainDataset(
            val_handler,  # history 소스는 동일 train
            hp.max_len,
            self.num_items,
            rng,
            sample_per_user=hp.num_train_sample_per_user,
            preference_event_weight=hp.preference_event_weight,
            next_items_window_size=hp.next_items_window_size,
            final_item_sampling_weight=hp.final_item_sampling_weight,
            final_item_score=hp.final_item_score,
            use_rank_learning=hp.use_rank_learning,
        )
        self._val_dataset = _NonEmptyEvalSubset(
            MultiInterestEvalDataset(val_handler, hp.max_len), val_positives
        )
        self._test_dataset = _NonEmptyEvalSubset(
            MultiInterestEvalDataset(test_handler, hp.max_len), test_positives
        )
        logger.info(
            f"[MI-v2] datasets ready — train __len__={len(self._train_dataset):,}, "
            f"val={len(self._val_dataset):,}, test={len(self._test_dataset):,}"
        )

    # ------------- helpers -------------

    @staticmethod
    def _load_interactions(files: List[Path]) -> pd.DataFrame:
        dfs = [
            pd.read_parquet(f, columns=["user_code", "goods_sno", "event", "ts"])
            for f in files
        ]
        df = pd.concat(dfs, ignore_index=True).dropna()
        df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce").astype("Int64")
        df["user_code"] = pd.to_numeric(df["user_code"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["goods_sno", "user_code"])
        df["goods_sno"] = df["goods_sno"].astype("int64")
        df["user_code"] = df["user_code"].astype("int64")
        return df

    def _build_category_mapping(self, item_meta_path: Path):
        item_meta = pd.read_parquet(
            item_meta_path, columns=["goods_sno", "standard_category_sno"]
        )
        item_meta["standard_category_sno"] = (
            pd.to_numeric(item_meta["standard_category_sno"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

        unique_cats = np.sort(item_meta["standard_category_sno"].unique())
        unique_cats = unique_cats[unique_cats > 0]
        cat_to_idx = {int(c): i + 1 for i, c in enumerate(unique_cats)}
        self.num_standard_categories = len(unique_cats) + 1  # PAD=0

        sno_to_cat = dict(
            zip(
                item_meta["goods_sno"].astype(int),
                item_meta["standard_category_sno"].astype(int),
            )
        )

        item_to_cat = np.zeros(self.num_items, dtype=np.int64)
        for sno, item_idx in self._sno_to_idx.items():
            item_to_cat[item_idx] = cat_to_idx.get(sno_to_cat.get(sno, 0), 0)

        self.item_feat_values = {
            FeatureField.ITEM_STANDARD_CATEGORY: item_to_cat,
        }
        logger.info(
            f"[MI-v2] standard categories: {self.num_standard_categories} (PAD=0)"
        )

    def _build_eval_positives(
        self,
        dir_path: Path,
        days: int,
        item_set: set,
        user_to_idx: Dict[int, int],
        num_users: int,
        max_users: Optional[int],
        split_label: str,
    ) -> List[np.ndarray]:
        files = sorted(dir_path.glob("*.parquet"))[-days:]
        df = self._load_interactions(files)
        df = df[df["goods_sno"].isin(item_set)]
        df["item_idx"] = df["goods_sno"].map(self._sno_to_idx).astype("int64")
        df = df.sort_values(["user_code", "ts"], kind="stable")

        positives: List[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(num_users)]
        for uc, g in df.groupby("user_code", sort=False):
            idx = user_to_idx.get(int(uc))
            if idx is None:
                continue
            positives[idx] = np.array([int(g["item_idx"].iloc[0])], dtype=np.int64)

        non_empty = sum(1 for p in positives if len(p) > 0)
        if max_users is not None and non_empty > max_users:
            rng = np.random.default_rng(self.hparams.seed)
            non_empty_idxs = np.array(
                [i for i, p in enumerate(positives) if len(p) > 0]
            )
            sel = set(rng.choice(non_empty_idxs, size=max_users, replace=False).tolist())
            positives = [p if i in sel else np.array([], dtype=np.int64) for i, p in enumerate(positives)]
            non_empty = max_users
        logger.info(f"[MI-v2] {split_label} eligible users with positive: {non_empty:,}")
        return positives

    # ------------- DataLoaders -------------

    def train_dataloader(self):
        return data_utils.DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            drop_last=self.hparams.drop_last_batch,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self._val_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self._test_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )
