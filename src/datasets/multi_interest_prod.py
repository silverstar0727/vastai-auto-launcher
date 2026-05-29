"""Production-matching Multi-Interest DataModule.

ably-reco production (`aws_code/reco_lib/reco_common/preprocess/default_raw_df_loader.py`)
의 전처리 흐름을 v2_full parquet 에 그대로 적용:

  0. Raw interactions 로드 (**train + val** = 105일 결합. test 16일은 봉인 →
     task #18 의 최종 비교 평가에서만 사용). production 의 1년치 학습 pool 흐름
     을 우리 보유 데이터(train+val) 한도 내에서 재현.
  1. on_sale 필터 (proxy: active_items.parquet 의 goods_sno).
  2. filter_short_seq(min_actions_per_user, count_unique=False) — 1st pass.
  2.5 preference_per_click 비율 필터 (suspicious item 제외).
  3. filter_popular_items(top max_items) — 인기 top-K item universe.
  4. filter_short_seq(min_actions_per_user, count_unique=True) — 2nd pass (★ 진짜
     engaged user 필터).

User split: per-user 시퀀스의 **마지막 1 item** → test, 나머지 → train.
test cap = max_test_samples (production default 200,000).

MultiInterestTrainDataset / MultiInterestEvalDataset 는 production 과 동일한 코드
(reco-lightning 의 multi_interest.py 가 1:1 migration).

기존 SIDTokenizerDataModule / MultiInterestV2DataModule 무변경.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
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


class _ProdDatasetHandler:
    """production 의 DatasetHandler 인터페이스 stand-in."""

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
        return pd.DataFrame({
            DatasetField.ITEM_INDEX: self.user_train_items[user_index],
            DatasetField.EVENT_CODE: self.user_train_events[user_index],
        })

    def get_items(self, user_index: int, split: str) -> list:
        if split == "test":
            return list(self.user_test_items[user_index])
        raise ValueError(f"unknown split: {split}")


class MultiInterestProdDataModule(L.LightningDataModule):
    """Production-identical Multi-Interest DataModule."""

    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        # production defaults (kubeflow override 반영)
        max_items: int = 300_000,
        min_actions_per_user: int = 100,
        preference_per_click_threshold: float = 0.02,
        min_click_threshold: int = 30_000,
        # 시퀀스
        max_len: int = 80,
        # 배치 (production 그대로)
        batch_size: int = 2048,
        test_batch_size: int = 64,
        num_workers: int = 4,
        drop_last_batch: bool = True,
        # 학습 샘플
        num_train_sample_per_user: int = 60,
        preference_event_weight: float = 40.0,
        next_items_window_size: int = 5,
        final_item_sampling_weight: float = 3.0,
        final_item_score: float = 1.0,
        use_rank_learning: bool = False,
        # test cap (production max_test_samples=200000)
        max_test_samples: int = 200_000,
        # production 데이터 크기 정합 — 필터 후 eligible user 가 이 값을 초과하면 sampling 으로 downsize
        # production 운영 로그: 1,128,409 users
        max_eligible_users: int = 1_128_409,
        # 학습 윈도우 — train+val pool 의 최근 N 일만 사용 (production ~60일 대비 보수적)
        last_n_days: int = 45,
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

    def setup(self, stage=None):
        if self._train_dataset is not None:
            return

        hp = self.hparams
        base = Path(hp.base_dir)

        # =========================================================
        # 0. interactions train+val pool 의 최근 N 일만 사용 (test 16일 봉인)
        #    production 은 ~60일 (last_n_days). 우리도 last N 일을 chronological 순으로 잘라 사용.
        # =========================================================
        pool = []
        for split_dir in ["train", "val"]:
            pool += list((base / "interactions" / split_dir).glob("*.parquet"))
        # 파일명이 YYYYMMDD.parquet 형식 → sort 가 곧 chronological order
        pool = sorted(pool, key=lambda p: p.name)
        all_files = pool[-hp.last_n_days:]
        print(
            f"[MI-prod] interaction files: {len(all_files)} "
            f"(train+val pool {len(pool)}일 중 최근 {hp.last_n_days}일, test 봉인)",
            flush=True,
        )
        if len(all_files) >= 1:
            print(
                f"[MI-prod]   range: {all_files[0].name} ~ {all_files[-1].name}",
                flush=True,
            )

        # ---- Pass A: item popularity (goods_sno 만 읽음) + event 통계 (pref/click 필터용)
        item_counter: Counter = Counter()
        item_click: Counter = Counter()
        item_pref: Counter = Counter()
        for i, f in enumerate(all_files):
            df = pd.read_parquet(f, columns=["goods_sno", "event"])
            df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce")
            df = df.dropna(subset=["goods_sno"])
            df["goods_sno"] = df["goods_sno"].astype("int64")
            df["mapped"] = df["event"].map(EVENT_MAP)
            snos = df["goods_sno"].to_numpy().tolist()
            mapped = df["mapped"].to_numpy().tolist()
            item_counter.update(snos)
            for sno, m in zip(snos, mapped):
                if m == "click":
                    item_click[sno] += 1
                elif m == "preference":
                    item_pref[sno] += 1
            if (i + 1) % 30 == 0 or i == len(all_files) - 1:
                print(f"[MI-prod] passA item-stats {i+1}/{len(all_files)} — unique items={len(item_counter):,}", flush=True)
            del df

        # =========================================================
        # 1. on_sale 필터 — active_items.parquet 의 goods_sno 사용 (proxy)
        # =========================================================
        active = pd.read_parquet(base / "meta" / "active_items.parquet", columns=["goods_sno"])
        active["goods_sno"] = pd.to_numeric(active["goods_sno"], errors="coerce").astype("int64")
        on_sale_set = set(active["goods_sno"].dropna().astype("int64").tolist())
        print(f"[MI-prod] step1 on_sale items: {len(on_sale_set):,}", flush=True)

        # =========================================================
        # 2.5 preference/click 비율 필터 — suspicious item 제외
        # =========================================================
        pref_per_click_deny = set()
        for sno in item_click:
            clicks = item_click[sno]
            prefs = item_pref.get(sno, 0)
            if clicks >= hp.min_click_threshold:
                ratio = prefs / clicks if clicks > 0 else 0
                if ratio > hp.preference_per_click_threshold:
                    pref_per_click_deny.add(sno)
        print(f"[MI-prod] step2.5 pref/click deny: {len(pref_per_click_deny):,}", flush=True)

        # =========================================================
        # 3. 인기 top-N — on_sale ∩ NOT deny 안에서 max_items
        # =========================================================
        eligible_items = on_sale_set - pref_per_click_deny
        # popularity 기준 top-N
        item_pop = [(s, c) for s, c in item_counter.items() if s in eligible_items]
        item_pop.sort(key=lambda x: -x[1])
        top_items = sorted([s for s, _ in item_pop[: hp.max_items]])
        item_set = set(top_items)
        self._sno_to_idx = {int(s): i + 1 for i, s in enumerate(top_items)}
        self.num_items = len(top_items) + 1
        print(f"[MI-prod] step3 popular top-{hp.max_items}: {len(top_items):,} (PAD=0, idx=1..{self.num_items-1})", flush=True)
        del item_counter, item_click, item_pref, item_pop

        # category 매핑
        self._build_category_mapping(base / "meta" / "item_meta.parquet")

        # =========================================================
        # Pass B: per-user item 누적 — count_unique=False 1st pass + 시퀀스 빌드
        # =========================================================
        user_items: Dict[str, List[int]] = {}    # user_code → [item_idx, ...]
        user_events: Dict[str, List[int]] = {}
        user_ts: Dict[str, List[int]] = {}
        for i, f in enumerate(all_files):
            df = pd.read_parquet(f, columns=["user_code", "goods_sno", "event", "ts"]).dropna(subset=["goods_sno", "user_code"])
            df["goods_sno"] = pd.to_numeric(df["goods_sno"], errors="coerce")
            df = df.dropna(subset=["goods_sno"])
            df["goods_sno"] = df["goods_sno"].astype("int64")
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0).astype("int64")
            df = df[df["goods_sno"].isin(item_set)]
            if len(df) == 0:
                del df; continue
            df["event_code"] = df["event"].map(EVENT_MAP).map(EVENT_VOCA).fillna(0).astype("int64")
            df["item_idx"] = df["goods_sno"].map(self._sno_to_idx).astype("int64")
            df = df.sort_values(["user_code", "ts"], kind="stable")
            for uc, g in df.groupby("user_code", sort=False):
                uc_key = str(uc)
                if uc_key not in user_items:
                    user_items[uc_key] = []
                    user_events[uc_key] = []
                    user_ts[uc_key] = []
                user_items[uc_key].extend(g["item_idx"].tolist())
                user_events[uc_key].extend(g["event_code"].tolist())
                user_ts[uc_key].extend(g["ts"].tolist())
            if (i + 1) % 30 == 0 or i == len(all_files) - 1:
                print(f"[MI-prod] passB per-user {i+1}/{len(all_files)} — users so far: {len(user_items):,}", flush=True)
            del df

        # =========================================================
        # 4. filter_short_seq count_unique=True (2nd pass) — ★ 진짜 engaged user
        # =========================================================
        user_train_items: List[np.ndarray] = []
        user_train_events: List[np.ndarray] = []
        user_test_items: List[np.ndarray] = []
        eligible_user_codes: List[str] = []
        rng = np.random.default_rng(hp.seed)
        for uc in sorted(user_items.keys()):
            items = user_items[uc]
            events = user_events[uc]
            n_unique = len(set(items))
            if n_unique < hp.min_actions_per_user:
                continue
            if len(items) < 2:
                continue
            # leave-last-out: 마지막 1 item → test
            test_item = items[-1]
            train_items = items[:-1]
            train_events = events[:-1]
            # max_len 보다 시퀀스가 길면 자르지 않고 그대로 (Train/EvalDataset 가 알아서 처리)
            user_train_items.append(np.asarray(train_items, dtype=np.int64))
            user_train_events.append(np.asarray(train_events, dtype=np.int64))
            user_test_items.append(np.asarray([test_item], dtype=np.int64))
            eligible_user_codes.append(uc)

        total_pre = len(user_items)
        del user_items, user_events, user_ts
        print(
            f"[MI-prod] step4 eligible users (unique items ≥ {hp.min_actions_per_user}): "
            f"{len(eligible_user_codes):,} / {total_pre:,}",
            flush=True,
        )

        # =========================================================
        # 데이터 크기 정합 — production user 수에 맞춰 downsize (필요 시)
        # =========================================================
        if len(eligible_user_codes) > hp.max_eligible_users:
            sel = rng.choice(
                len(eligible_user_codes), size=hp.max_eligible_users, replace=False
            )
            sel = sorted(sel.tolist())
            user_train_items = [user_train_items[i] for i in sel]
            user_train_events = [user_train_events[i] for i in sel]
            user_test_items = [user_test_items[i] for i in sel]
            eligible_user_codes = [eligible_user_codes[i] for i in sel]
            print(
                f"[MI-prod] downsized to production user count: "
                f"{len(eligible_user_codes):,} (target={hp.max_eligible_users:,})",
                flush=True,
            )
        else:
            print(
                f"[MI-prod] eligible {len(eligible_user_codes):,} ≤ "
                f"target {hp.max_eligible_users:,} → 전체 사용",
                flush=True,
            )

        # =========================================================
        # test sample cap — production max_test_samples (예: 200,000)
        # train 은 그대로 유지, eval set 만 cap
        # =========================================================
        n_total = len(user_test_items)
        if n_total > hp.max_test_samples:
            # 랜덤 sample 선택
            sel = rng.choice(n_total, size=hp.max_test_samples, replace=False)
            eval_indices = sorted(sel.tolist())
        else:
            eval_indices = list(range(n_total))
        print(
            f"[MI-prod] eval cap: {len(eval_indices):,} / {n_total:,} (max_test_samples={hp.max_test_samples:,})",
            flush=True,
        )

        # =========================================================
        # Dataset 생성 — train 은 전체 user, eval 은 cap 된 subset
        # =========================================================
        handler = _ProdDatasetHandler(user_train_items, user_train_events, user_test_items)
        rng_py = random.Random(int(hp.dataloader_random_seed))

        self._train_dataset = MultiInterestTrainDataset(
            handler,
            hp.max_len,
            self.num_items,
            rng_py,
            sample_per_user=hp.num_train_sample_per_user,
            preference_event_weight=hp.preference_event_weight,
            next_items_window_size=hp.next_items_window_size,
            final_item_sampling_weight=hp.final_item_sampling_weight,
            final_item_score=hp.final_item_score,
            use_rank_learning=hp.use_rank_learning,
        )
        # eval 은 capped subset 만 사용 — base eval dataset 을 wrap
        self._val_dataset = _ProdEvalSubset(
            MultiInterestEvalDataset(handler, hp.max_len), eval_indices
        )
        self._test_dataset = self._val_dataset  # production 은 단일 test set 사용
        print(
            f"[MI-prod] datasets ready — train __len__={len(self._train_dataset):,}, "
            f"eval={len(self._val_dataset):,}",
            flush=True,
        )

    # ------------- helpers -------------

    def _build_category_mapping(self, item_meta_path: Path):
        item_meta = pd.read_parquet(item_meta_path, columns=["goods_sno", "standard_category_sno"])
        item_meta["standard_category_sno"] = (
            pd.to_numeric(item_meta["standard_category_sno"], errors="coerce").fillna(0).astype("int64")
        )
        unique_cats = np.sort(item_meta["standard_category_sno"].unique())
        unique_cats = unique_cats[unique_cats > 0]
        cat_to_idx = {int(c): i + 1 for i, c in enumerate(unique_cats)}
        self.num_standard_categories = len(unique_cats) + 1
        sno_to_cat = dict(
            zip(item_meta["goods_sno"].astype(int), item_meta["standard_category_sno"].astype(int))
        )
        item_to_cat = np.zeros(self.num_items, dtype=np.int64)
        for sno, item_idx in self._sno_to_idx.items():
            item_to_cat[item_idx] = cat_to_idx.get(sno_to_cat.get(sno, 0), 0)
        self.item_feat_values = {FeatureField.ITEM_STANDARD_CATEGORY: item_to_cat}
        print(f"[MI-prod] standard categories: {self.num_standard_categories}", flush=True)

    # ------------- DataLoaders -------------

    def train_dataloader(self):
        return data_utils.DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=self.hparams.drop_last_batch,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self._val_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class _ProdEvalSubset(data_utils.Dataset):
    def __init__(self, base_eval: MultiInterestEvalDataset, indices: List[int]):
        self.base = base_eval
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
