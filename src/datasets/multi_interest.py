"""Multi-Interest DataModule.

기존 multi_interest/train/pl_factory.py + dataloaders.py를
Lightning DataModule 패턴으로 통합.

CBFDataModule과의 핵심 차이점:
1. Dataset: EVENT_CODE를 포함하는 시퀀스 예측 태스크 (next-item prediction)
2. Split: seq_len=max_len*2, random_seed=None (마지막 아이템으로 테스트)
3. 학습 샘플: sample_per_user 만큼 유저당 복수 샘플 생성
4. preference_event_weight: 선호 이벤트에 가중치를 부여하여 학습 위치 샘플링
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

from utils.constants import EVENT_VOCA, DatasetField, FeatureField, RawDataField
from utils.data_handler import DatasetHandler, split_data_by_users
from utils.goods_info import GoodsInfo
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix
from utils.preprocess import PreprocessResult, default_load_raw_df_items

logger = logging.getLogger(__name__)


# --- Constants ---


class ModelInputKey:
    POSITIVE_ITEMS = "positive_items"
    LABEL_ITEMS = "label_items"
    LABEL_SCORES = "label_scores"


def _get_on_sale_items(goods_filename):
    """판매중인 아이템 목록 반환."""
    df_items = pd.read_csv(goods_filename, escapechar="\\")
    df_items = df_items.dropna()
    return list(df_items["sno"])


# --- Dataset Classes ---


class MultiInterestTrainDataset(data_utils.Dataset):
    """Multi-Interest 학습용 데이터셋.

    유저 히스토리에서 예측 위치를 랜덤 샘플링하여, 해당 위치 이전의 히스토리를 입력으로,
    이후의 아이템을 레이블로 사용한다.

    Args:
        sample_per_user: 유저당 학습 샘플 수
        preference_event_weight: 선호 이벤트(찜/구매)에 부여할 가중치
        next_items_window_size: 예측 대상 아이템 윈도우 크기 (1이면 next-item prediction)
        final_item_sampling_weight: 윈도우 내 선호 이벤트 아이템에 부여할 샘플링 가중치
        final_item_score: 선호 이벤트 아이템의 레이블 점수 (rank learning 시 사용)
        use_rank_learning: 여러 상품 간 순위 학습 여부
    """

    def __init__(
        self,
        dataset_handler: DatasetHandler,
        max_len: int,
        num_items: int,
        rng: random.Random,
        sample_per_user: int = 1,
        preference_event_weight: float = 1.0,
        next_items_window_size: int = 1,
        final_item_sampling_weight: float = 3.0,
        final_item_score: float = 1.0,
        use_rank_learning: bool = False,
    ):
        self.dataset_handler = dataset_handler
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.sample_per_user = sample_per_user
        self.preference_event_weight = preference_event_weight
        self.next_items_window_size = next_items_window_size
        self.final_item_sampling_weight = final_item_sampling_weight
        self.final_item_score = final_item_score
        self.use_rank_learning = use_rank_learning

    def __len__(self):
        return self.dataset_handler.num_train_samples() * self.sample_per_user

    def __getitem__(self, index):
        user_index = int(index / self.sample_per_user)
        train_df = self.dataset_handler.get_user2items(user_index, "train")

        items = train_df[DatasetField.ITEM_INDEX].tolist()
        events = train_df[DatasetField.EVENT_CODE].tolist()

        assert len(items) > (self.next_items_window_size + 1), (
            f"history is shorter than window_size. len(items): {len(items)}"
        )

        # 유저 히스토리에서 예측할 위치를 랜덤으로 선택 (preference event 가중치 반영)
        if self.next_items_window_size > 1:
            candidate_events = events[1 : 1 - self.next_items_window_size]
        else:
            candidate_events = events[1:]
        position = 1 + self._get_event_sample_index(
            candidate_events,
            preference_event_weight=self.preference_event_weight,
        )

        # position 위치 이후로 액션이 발생한 상품들 중에서 하나를 샘플링
        next_items = items[position : position + self.next_items_window_size]
        next_events = events[position : position + self.next_items_window_size]
        next_scores = [
            self.final_item_score if x == EVENT_VOCA["preference"] else 1.0
            for x in next_events
        ]
        positive_item_index = self._get_event_sample_index(
            events=next_events,
            preference_event_weight=self.final_item_sampling_weight,
        )
        positive_item = next_items[positive_item_index]

        history_items = items[:position]
        history_events = events[:position]
        if len(history_items) > self.max_len:
            history_items = history_items[-self.max_len :]
            history_events = history_events[-self.max_len :]
        pad_len = self.max_len - len(history_items)
        if pad_len > 0:
            history_items = [0] * pad_len + history_items  # left padding
            history_events = [0] * pad_len + history_events

        input_tensors = {
            FeatureField.CLICK_ITEMS: torch.LongTensor(history_items),
            FeatureField.EVENT_CODE: torch.LongTensor(history_events),
            ModelInputKey.POSITIVE_ITEMS: torch.LongTensor([positive_item]),
        }
        if self.use_rank_learning:
            input_tensors[ModelInputKey.LABEL_ITEMS] = torch.LongTensor(next_items)
            input_tensors[ModelInputKey.LABEL_SCORES] = torch.FloatTensor(next_scores)
        return input_tensors

    def _get_event_sample_index(
        self,
        events: list,
        preference_event_weight: float,
    ) -> int:
        if len(events) <= 1:
            return 0
        weights = [
            preference_event_weight if x == EVENT_VOCA["preference"] else 1.0
            for x in events
        ]
        return self.rng.choices(range(len(weights)), weights=weights)[0]


class MultiInterestEvalDataset(data_utils.Dataset):
    """Multi-Interest 평가용 데이터셋.

    전체 히스토리를 입력으로 사용하고, 테스트 데이터의 첫 번째 상품만 평가에 사용한다.
    """

    def __init__(self, dataset_handler: DatasetHandler, max_len: int):
        self.dataset_handler = dataset_handler
        self.max_len = max_len

    def __len__(self):
        return self.dataset_handler.num_test_samples()

    def __getitem__(self, index):
        train_df = self.dataset_handler.get_user2items(index, "train")
        items = train_df[DatasetField.ITEM_INDEX].tolist()
        events = train_df[DatasetField.EVENT_CODE].tolist()

        if len(items) > self.max_len:
            items = items[-self.max_len :]
            events = events[-self.max_len :]
        pad_len = self.max_len - len(items)
        if pad_len > 0:
            items = [0] * pad_len + items
            events = [0] * pad_len + events

        positive_items = self.dataset_handler.get_items(index, "test")

        input_dict = {
            FeatureField.CLICK_ITEMS: torch.LongTensor(items),
            FeatureField.EVENT_CODE: torch.LongTensor(events),
        }
        # 테스트 데이터의 첫 번째 상품만 평가에 사용
        return input_dict, torch.LongTensor(positive_items[:1])


# --- DataModule ---


class MultiInterestDataModule(L.LightningDataModule):
    def __init__(
        self,
        # 데이터 경로
        raw_dataset_root: str = "",
        sql_dataset_root: str = "",
        model_path: str = "",
        goods_filename: str = "",
        category_filename: str = "",
        standard_category_filename: str = "",
        user_filename: str = "",
        # 데이터 필터링
        min_actions_per_user: int = 100,
        max_items: int = 300000,
        # 배치 설정
        batch_size: int = 2048,
        test_batch_size: int = 64,
        max_len: int = 80,
        num_workers: int = 2,
        drop_last_batch: bool = True,
        # 학습 샘플 설정
        num_train_sample_per_user: int = 60,
        preference_event_weight: float = 40.0,
        next_items_window_size: int = 5,
        final_item_sampling_weight: float = 3.0,
        final_item_score: float = 1.0,
        use_rank_learning: bool = False,
        # 아이템 학습 가중치
        use_item_train_weight: bool = False,
        goods_train_weight_root: str = "",
        # 데이터 분할
        max_test_samples: int = 200000,
        num_test_samples_per_user: int = 1,
        # 시드
        dataloader_random_seed: float = 0.0,
        # preference/click 필터
        preference_per_click_condition: Optional[Dict] = None,
        # 기타
        save_preprocessed_data: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # setup()에서 채워질 속성 (Model에서 읽어감)
        self.num_items: Optional[int] = None
        self.num_standard_categories: Optional[int] = None
        self.item_feat_values: Optional[Dict[str, np.ndarray]] = None
        self.item_train_weights: Optional[np.ndarray] = None
        self.goods_info: Optional[GoodsInfo] = None

        self._train_dataset = None
        self._eval_dataset = None

    def setup(self, stage=None):
        if self._train_dataset is not None:
            return

        hp = self.hparams

        # 1. 전처리: CSV 로드 → label encoder 생성 → index 매핑
        preprocess_result = self._run_preprocess()

        # 2. GoodsInfo 생성
        self.goods_info = GoodsInfo(preprocess_result.df_items)

        # 3. item side features (ITEM_STANDARD_CATEGORY만 사용)
        item_side_features = self.goods_info.get_item_side_features()
        self.item_feat_values = {
            FeatureField.ITEM_STANDARD_CATEGORY: item_side_features[FeatureField.ITEM_STANDARD_CATEGORY],
        }

        # 4. num_standard_categories (label encoder에서 가져옴)
        self.num_standard_categories = (
            LabelEncoder.from_model_dir(hp.model_path, LabelEncoderPrefix.STANDARD_CATEGORY).get_valid_num_items() + 1
        )

        # 5. num_items (모델의 embedding vocab = goods_info items + 1 for padding)
        self.num_items = self.goods_info.get_valid_num_items() + 1

        # 6. 아이템 학습 가중치 (옵션)
        if hp.use_item_train_weight and hp.goods_train_weight_root:
            self.item_train_weights = self._read_item_train_weight()

        # 7. 유저 분할
        feature_names = [FeatureField.CLICK_ITEMS, FeatureField.EVENT_CODE]
        dataset_handler = self._load_dataset_handler(preprocess_result, feature_names)

        # 8. Dataset 생성
        rng = random.Random(hp.dataloader_random_seed)

        self._train_dataset = MultiInterestTrainDataset(
            dataset_handler,
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
        self._eval_dataset = MultiInterestEvalDataset(dataset_handler, hp.max_len)

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
            self._eval_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    # --- Private methods ---

    def _run_preprocess(self) -> PreprocessResult:
        """기존 PytorchLightningFactory.preprocess() + DefaultRawDfLoader 로직."""
        hp = self.hparams
        model_path = Path(hp.model_path)
        preprocessed_root = model_path.joinpath("preprocessed")

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(preprocessed_root, exist_ok=True)

        goods_filename = hp.goods_filename
        category_filename = hp.category_filename
        standard_category_filename = hp.standard_category_filename

        def load_user2items():
            return self._load_raw_user2items()

        def load_raw_df_items(on_voca_items):
            return default_load_raw_df_items(
                goods_filename=goods_filename,
                category_filename=category_filename,
                standard_category_filename=standard_category_filename,
                on_voca_items=on_voca_items,
            )

        from utils.preprocess import preprocess

        user_file_path = (
            Path(hp.user_filename) if hp.user_filename else Path(hp.sql_dataset_root) / "member.csv"
        )

        return preprocess(
            load_user2items=load_user2items,
            load_raw_df_items=load_raw_df_items,
            preprocessed_root=preprocessed_root,
            model_path=model_path,
            user_file_path=user_file_path,
            use_search_data=False,
            save_preprocessed_data=hp.save_preprocessed_data,
            save_goods_info=hp.use_item_train_weight,
        )

    def _load_raw_user2items(self):
        """DefaultRawDfLoader.load_raw_user2items()와 동일한 구현.

        CBFDataModule._load_raw_user2items()와 동일한 필터링 로직 (EVENT_MAP 사용).
        """
        hp = self.hparams
        from utils.df_utils import (
            filter_popular_items,
            filter_short_seq,
            find_items_by_preference_per_click,
            load_df_user2items,
        )

        interaction_filepath = Path(hp.raw_dataset_root)
        order_filepath = Path(hp.sql_dataset_root) / "order.csv"
        goods_filepath = hp.goods_filename

        min_actions_per_user = hp.min_actions_per_user
        item_voca_size = hp.max_items
        preference_per_click_condition = hp.preference_per_click_condition
        if preference_per_click_condition is None:
            preference_per_click_condition = {
                "min_click_threshold": 30000,
                "preference_per_click_threshold": 0.02,
            }

        # 1. 인터랙션 + 주문 데이터 로드
        df = load_df_user2items(interaction_filepath, order_filepath)

        # on-sale 아이템 필터
        on_sale_items = _get_on_sale_items(goods_filepath)
        logger.info(f"0. Raw : # of interactions {len(df)}")

        df = df[df[RawDataField.ITEM_ID].isin(on_sale_items)]
        logger.info(f"1. on sale filter: # of interactions {len(df)}")

        # 2. min_actions_per_user 필터
        df = filter_short_seq(df, min_actions_per_user, count_unique=False)
        logger.info(f"2. min user filter : # of interactions {len(df)}")

        # 2.5 preference/click 비율 필터
        pref_per_click_deny_items = find_items_by_preference_per_click(
            df,
            preference_per_click_condition,
        )
        logger.info(
            f"2.5 pref_per_click_deny_items({len(pref_per_click_deny_items)}) : "
            f"{pref_per_click_deny_items[:100]}"
        )
        df = df[~df[RawDataField.ITEM_ID].isin(pref_per_click_deny_items)]

        # 3. max_voca_items 필터
        on_voca_items = filter_popular_items(df, item_voca_size)

        df = df[df[RawDataField.ITEM_ID].isin(on_voca_items)]
        logger.info(f"3. on vocabulary filter: # of interactions {len(df)}")

        # 4. vocabulary 축소 후 유저 재필터
        df = filter_short_seq(df, min_actions_per_user, count_unique=True)
        logger.info(f"4. min user filter : # of interactions {len(df)}")

        return df, on_voca_items

    def _load_dataset_handler(
        self, preprocess_result: PreprocessResult, feature_names: List[str]
    ) -> DatasetHandler:
        """기존 basic_pl.py의 _load_dataset_handler() 로직.

        CBFDataModule과의 차이점:
        - seq_len = max_len * 2 (긴 시퀀스를 분할)
        - random_seed = None (마지막 아이템으로 테스트, 랜덤이 아님)
        - num_test_samples_per_user = 1 (기본값)
        """
        hp = self.hparams
        model_path = Path(hp.model_path)
        preprocessed_root = model_path.joinpath("preprocessed")
        dataset_path = preprocessed_root.joinpath("dataset.pkl")

        if dataset_path.is_file():
            logger.info("Already User splitting preprocessed. Skip preprocessing")
            dataset_handler = DatasetHandler.from_pickle(dataset_path)
        else:
            if not dataset_path.parent.is_dir():
                dataset_path.parent.mkdir(parents=True)

            split_params = {
                "max_seq_per_user": None,
                "seq_len": hp.max_len * 2,
                "random_seed": None,
                "n_test_samples_per_seq": hp.num_test_samples_per_user,
                "remove_duplicate_in_seq": True,
            }

            dataset_handler = split_data_by_users(
                preprocess_result.df_user2items,
                preprocess_result.users_static,
                feature_names,
                max_test_samples=hp.max_test_samples,
                **split_params,
            )
            if hp.save_preprocessed_data:
                dataset_handler.save(dataset_path)
                logger.info(f"dataset_handler saved {dataset_path}")
        return dataset_handler

    def _read_item_train_weight(self) -> np.ndarray:
        """상품별 학습 가중치를 파일에서 읽어서 np.array로 리턴한다."""
        hp = self.hparams
        import glob as glob_module

        df_goods = pd.read_pickle(os.path.join(hp.model_path, "df_goods.pkl"))[
            [RawDataField.ITEM_ID, DatasetField.ITEM_INDEX]
        ]

        csv_files = glob_module.glob(
            os.path.join(hp.goods_train_weight_root, "goods_train_weight/*.csv")
        )
        df_goods_weight = pd.concat(
            [pd.read_csv(f, sep=",") for f in csv_files], ignore_index=True
        ).rename(columns={"goods_sno": RawDataField.ITEM_ID})[
            [RawDataField.ITEM_ID, "goods_train_weight"]
        ]

        df_item_weight = pd.merge(df_goods, df_goods_weight, on=RawDataField.ITEM_ID)[
            [DatasetField.ITEM_INDEX, "goods_train_weight"]
        ]

        assert df_item_weight.shape[0] == df_goods.shape[0], (
            "df_item_weight.shape[0] != df_goods.shape[0]"
        )

        num_items = self.goods_info.get_valid_num_items()
        item_train_weights = np.zeros(num_items + 1)
        item_train_weights[df_item_weight[DatasetField.ITEM_INDEX]] = df_item_weight[
            "goods_train_weight"
        ]
        return item_train_weights
