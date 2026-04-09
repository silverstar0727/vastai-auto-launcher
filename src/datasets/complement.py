"""Complement DataModule.

기존 complement/train/pl_factory.py + dataloaders.py + raw_df_loader.py를
Lightning DataModule 패턴으로 통합.

CBFDataModule과의 핵심 차이점:
1. _load_raw_user2items: COMPLEMENT_EVENT_MAP 사용, 분리 vocabulary (interaction + cart/order)
2. _get_feature_process_spec: ITEM_CATEGORY 사용 (ITEM_STANDARD_CATEGORY 대신)
3. _get_item_side_feature: get_item_side_features() 사용 (모든 side feature 포함)
4. use_attr 지원 (attribute feature)
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

from nets.cbf.feature_layers import (
    Feature,
    FeatureInputLayerName,
    PolynomialFeat,
    SparseFeat,
)
from utils.constants import (
    COMPLEMENT_EVENT_MAP,
    EVENT_MAP,
    DatasetField,
    FeatureField,
    RawDataField,
)
from utils.data_handler import DatasetHandler, split_data_by_users
from utils.goods_info import GoodsInfo
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix
from utils.preprocess import PreprocessResult, default_load_raw_df_items

logger = logging.getLogger(__name__)


def _get_on_sale_items(goods_filename):
    df_items = pd.read_csv(goods_filename, escapechar="\\")
    df_items = df_items.dropna()
    return list(df_items["sno"])


# --- Dataset Classes (원본 complement dataloaders.py 그대로) ---


def _insert_pad(seq, max_len):
    pad_len = max_len - len(seq)
    seq_padded = [0] * pad_len + seq
    seq_padded = seq_padded[-max_len:]
    return seq_padded


class ComplementTrainDataset(data_utils.Dataset):
    def __init__(
        self,
        dataset_handler: DatasetHandler,
        max_len: int,
        num_items: int,
        rng: random.Random,
        max_pos_items: int = 400,
        use_negative_sampling: bool = True,
    ):
        self.dataset_handler = dataset_handler
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.max_pos_items = max_pos_items
        self.use_negative_sampling = use_negative_sampling

    def __len__(self):
        return self.dataset_handler.num_train_samples()

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        df = self.dataset_handler.get_user2items(index, "train")
        items = list(df[DatasetField.ITEM_INDEX])

        n_input_samples = min(int(len(items) * 0.5), self.max_len)
        input_items = self.rng.sample(items, n_input_samples)
        output_items = np.setdiff1d(items, input_items)

        input_items = _insert_pad(input_items, self.max_len)

        input_tensors = {
            FeatureField.CLICK_ITEMS: torch.LongTensor(input_items),
        }
        input_tensors.update(self.dataset_handler.get_user_info(index))

        if self.use_negative_sampling:
            if len(output_items) > self.max_pos_items:
                pos_labels = self.rng.sample(output_items.tolist(), self.max_pos_items)
            else:
                pos_labels = _insert_pad(output_items.tolist(), self.max_pos_items)
            return input_tensors, torch.LongTensor(pos_labels)
        else:
            y = np.zeros((self.num_items + 1))
            y[output_items] = 1.0
            return input_tensors, torch.FloatTensor(y)


class ComplementEvalDataset(data_utils.Dataset):
    def __init__(self, dataset_handler: DatasetHandler, max_len: int):
        self.dataset_handler = dataset_handler
        self.max_len = max_len

    def __len__(self):
        return self.dataset_handler.num_test_samples()

    def __getitem__(self, index) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        df = self.dataset_handler.get_user2items(index, "train")
        items = list(df[DatasetField.ITEM_INDEX])
        items = _insert_pad(items, self.max_len)
        input_tensors = {
            FeatureField.CLICK_ITEMS: torch.LongTensor(items),
        }
        input_tensors.update(self.dataset_handler.get_user_info(index))

        positive_items = self.dataset_handler.get_items(index, "test")
        return input_tensors, torch.LongTensor(positive_items)


# --- DataModule ---


class ComplementDataModule(L.LightningDataModule):
    def __init__(
        self,
        # 데이터 경로
        raw_dataset_root: str = "",
        pretrained_root: str = "",
        sql_dataset_root: str = "",
        model_path: str = "",
        goods_filename: str = "",
        category_filename: str = "",
        standard_category_filename: str = "",
        user_filename: str = "",
        # 데이터 필터링
        min_actions_per_user: int = 50,
        max_items: int = 600000,
        max_interaction_items: int = 180000,
        max_ordered_items: int = 400000,
        # 배치 설정
        batch_size: int = 256,
        test_batch_size: int = 256,
        max_len: int = 100,
        num_workers: int = 0,
        drop_last_batch: bool = True,
        # 피처 설정
        item_embed_size: int = 256,
        item_vocabulary_size: int = 600000,
        use_age_info: bool = True,
        use_in_house_text_embed: bool = True,
        use_item_emb_in_item_tower: bool = True,
        use_attr: bool = True,
        attr_meta_filename: str = "",
        attr_filename: str = "",
        market_embed_size: int = 64,
        category_embed_size: int = 64,
        # 학습 설정
        negative_sampling: bool = False,
        # 데이터 분할
        max_test_samples: int = 200000,
        num_test_samples_per_user: int = 10,
        # 시드
        dataloader_random_seed: float = 0.0,
        # deny condition
        deny_condition_min_click: int = 30000,
        deny_condition_events_per_click: float = 0.02,
        # 기타
        save_preprocessed_data: bool = True,
        use_search_data: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # setup()에서 채워질 속성
        self.features: Optional[List[Feature]] = None
        self.user_feat_names: Optional[List[str]] = None
        self.item_feat_names: Optional[List[str]] = None
        self.item_feat_values: Optional[Dict[str, np.array]] = None
        self.num_items: Optional[int] = None
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

        # 3. feature spec 구성
        features, user_feat_names, item_feat_names = self._get_feature_process_spec(self.goods_info)
        self.features = features
        self.user_feat_names = user_feat_names
        self.item_feat_names = item_feat_names

        # 4. item side features (complement는 모든 side feature 사용)
        self.item_feat_values = self.goods_info.get_item_side_features()

        # 5. num_items
        self.num_items = self.goods_info.get_valid_num_items() + 1

        # 6. 유저 분할
        model_feature_names = list(set(user_feat_names + item_feat_names))
        dataset_handler = self._load_dataset_handler(
            preprocess_result, model_feature_names
        )

        # 7. Dataset 생성
        dataset_num_items = preprocess_result.get_num_vocab_items()
        rng = random.Random(hp.dataloader_random_seed)

        self._train_dataset = ComplementTrainDataset(
            dataset_handler, hp.max_len, dataset_num_items, rng,
            use_negative_sampling=hp.negative_sampling,
        )
        self._eval_dataset = ComplementEvalDataset(dataset_handler, hp.max_len)

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
        hp = self.hparams
        raw_dataset_root = Path(hp.raw_dataset_root)
        model_path = Path(hp.model_path)
        preprocessed_root = raw_dataset_root.joinpath("preprocessed")

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
        user_file_path = Path(hp.user_filename) if hp.user_filename else Path(hp.sql_dataset_root) / "member.csv"

        return preprocess(
            load_user2items=load_user2items,
            load_raw_df_items=load_raw_df_items,
            preprocessed_root=preprocessed_root,
            model_path=model_path,
            user_file_path=user_file_path,
            use_search_data=hp.use_search_data,
            save_preprocessed_data=hp.save_preprocessed_data,
        )

    def _load_raw_user2items(self):
        """ComplementRawDfLoader.load_raw_user2items()와 완전히 동일한 구현.

        CBFDataModule._load_raw_user2items()와의 차이점:
        1. COMPLEMENT_EVENT_MAP 사용 (개별 이벤트 타입 보존)
        2. deny_condition: order/cart/like 개별 이벤트로 비율 계산
        3. 분리 vocabulary: interaction items (max_interaction_items) + cart/order items (max_ordered_items)
        4. 최종 EVENT_MAP 매핑 (like/cart/order → preference)
        """
        hp = self.hparams
        from utils.df_utils import (
            filter_popular_items,
            filter_short_seq,
            load_df_user2items,
        )

        interaction_filepath = Path(hp.raw_dataset_root)
        order_filepath = Path(hp.sql_dataset_root) / "order.csv"
        goods_filepath = hp.goods_filename

        min_actions_per_user = hp.min_actions_per_user
        interaction_item_voca_size = hp.max_interaction_items
        order_item_voca_size = hp.max_ordered_items

        deny_condition = {
            "min_click_threshold": hp.deny_condition_min_click,
            "events_per_click_threshold": hp.deny_condition_events_per_click,
        }

        logger.info("start load user2items for complement reco")

        # 1. COMPLEMENT_EVENT_MAP으로 로드 (이벤트 타입 개별 보존)
        df = load_df_user2items(interaction_filepath, order_filepath, event_map=COMPLEMENT_EVENT_MAP)
        logger.info(f"0. Raw : # of interactions {len(df)}")

        # 2. on-sale 아이템 필터
        on_sale_items = _get_on_sale_items(goods_filepath)
        df = df[df[RawDataField.ITEM_ID].isin(on_sale_items)]
        logger.info(f"1. on sale filter: # of interactions {len(df)}")

        # 3. min_actions_per_user 필터
        df = filter_short_seq(df, min_actions_per_user, count_unique=False)
        logger.info(f"2. min user filter :  # of interactions {len(df)}")

        # 4. deny condition: 클릭은 많은데 찜/구매가 적은 상품 제외
        denied_items = self._find_deny_items_by_condition(df, deny_condition)
        logger.info(f"2.5 pref_per_click_deny_items({len(denied_items)}) : {denied_items[:100]}")
        df = df[~df[RawDataField.ITEM_ID].isin(denied_items)]

        # 5. interaction items vocabulary (기존 방식)
        on_voca_interaction_items = filter_popular_items(df, interaction_item_voca_size)
        logger.info(f"on_voca_interaction_items: {len(on_voca_interaction_items)}")

        # 6. cart/order items vocabulary (추가 vocabulary)
        order_and_cart_df = df[df[RawDataField.EVENT_TYPE].isin({"order", "cart"})]
        on_voca_cart_order_items = filter_popular_items(order_and_cart_df, max_items=order_item_voca_size)
        logger.info(f"on_voca_cart_order_items: {len(on_voca_cart_order_items)}")

        # 7. 두 vocabulary 합치기
        total_on_voca_items = set(on_voca_interaction_items).union(set(on_voca_cart_order_items))
        logger.info(f"total_on_voca_items: {len(total_on_voca_items)}")

        # 8. vocabulary에 포함된 아이템만 필터
        df = df[df[RawDataField.ITEM_ID].isin(total_on_voca_items)]
        logger.info(f"3. on vocabulary filter: # of interactions {len(df)}")

        # 9. vocabulary 축소 후 유저 재필터
        df = filter_short_seq(df, min_actions_per_user, count_unique=True)
        logger.info(f"4. min user filter :  # of interactions {len(df)}")

        # 10. EVENT_MAP으로 매핑 (like/cart/order → preference)
        df[RawDataField.EVENT_TYPE] = df[RawDataField.EVENT_TYPE].map(EVENT_MAP)

        return df, list(total_on_voca_items)

    @staticmethod
    def _find_deny_items_by_condition(df_input, condition):
        """클릭은 많은데 찜/구매 등의 액션 비율이 낮은 상품 리스트를 리턴한다.

        CBF의 find_items_by_preference_per_click과 달리 개별 이벤트 타입(order, cart, like)으로 계산.
        """
        target_event_types = ["order", "cart", "like"]

        try:
            df_item_action_stat = (
                df_input.groupby([RawDataField.ITEM_ID, RawDataField.EVENT_TYPE])
                .agg(cnt=(RawDataField.USER_ID, len))
                .reset_index()
                .pivot(index=RawDataField.ITEM_ID, columns=RawDataField.EVENT_TYPE, values="cnt")
                .fillna({"click": 0.0, "order": 0.0, "cart": 0.0, "like": 0.0})
                .reset_index()
                .assign(events_per_click=lambda x: x[target_event_types].sum(axis=1) / (1.0 + x["click"]))[
                    [RawDataField.ITEM_ID, "click", "events_per_click"]
                ]
            )
        except Exception:
            return []

        found_items = list(
            df_item_action_stat.query(f'click >= {condition["min_click_threshold"]}').query(
                f'events_per_click < {condition["events_per_click_threshold"]}'
            )[RawDataField.ITEM_ID]
        )
        return found_items

    def _load_dataset_handler(self, preprocess_result: PreprocessResult, feature_names: List[str]) -> DatasetHandler:
        hp = self.hparams
        raw_dataset_root = Path(hp.raw_dataset_root)
        preprocessed_root = raw_dataset_root.joinpath("preprocessed")
        dataset_path = preprocessed_root.joinpath("dataset.pkl")

        if dataset_path.is_file():
            logger.info("Already User splitting preprocessed. Skip preprocessing")
            dataset_handler = DatasetHandler.from_pickle(dataset_path)
        else:
            if not dataset_path.parent.is_dir():
                dataset_path.parent.mkdir(parents=True)

            split_params = {
                "max_seq_per_user": None,
                "seq_len": None,
                "random_seed": "111",
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

    def _get_feature_process_spec(self, goods_info: GoodsInfo) -> Tuple[List[Feature], List[str], List[str]]:
        """complement의 feature spec. CBF와의 차이: ITEM_CATEGORY 사용, use_attr 지원."""
        hp = self.hparams

        if hp.use_in_house_text_embed:
            pretrained_text_embed = goods_info.get_token_embeddings_arr(
                hp.pretrained_root,
                append_padding_index=True,
                num_special_index=0,
                do_normalize=False,
            )
        else:
            pretrained_text_embed = goods_info.get_text_embeddings_arr()

        vocabulary_size = hp.item_vocabulary_size + 1

        user_item_embed = SparseFeat(
            FeatureField.CLICK_ITEMS,
            vocabulary_size=vocabulary_size,
            embedding_dim=hp.item_embed_size,
            layer_name=FeatureInputLayerName.ITEM_EMBEDDING,
            padding_idx=0,
        )
        user_text_embed = SparseFeat(
            FeatureField.CLICK_ITEMS,
            vocabulary_size=pretrained_text_embed.shape[0],
            embedding_dim=pretrained_text_embed.shape[1],
            layer_name=FeatureInputLayerName.TEXT_EMBEDDING,
            embeddings_initializer=pretrained_text_embed,
        )
        item_text_embed = SparseFeat(
            FeatureField.ITEM,
            vocabulary_size=pretrained_text_embed.shape[0],
            embedding_dim=pretrained_text_embed.shape[1],
            layer_name=FeatureInputLayerName.TEXT_EMBEDDING,
            embeddings_initializer=pretrained_text_embed,
        )

        item_embed = SparseFeat(
            FeatureField.ITEM_DROPOUT,
            vocabulary_size=vocabulary_size,
            embedding_dim=hp.item_embed_size,
            layer_name=FeatureInputLayerName.ITEM_EMBEDDING,
            padding_idx=0,
        )
        user_features = [user_item_embed, user_text_embed]
        item_features = [item_text_embed]

        if hp.use_item_emb_in_item_tower:
            item_features += [item_embed]

        if hp.use_attr and hp.attr_meta_filename and hp.attr_filename:
            from utils.attribute import build_attr_feat
            attr_feat, user_attr_feat = build_attr_feat(
                hp.model_path,
                goods_info.df.sort_values(DatasetField.ITEM_INDEX),
                hp.attr_meta_filename,
                hp.attr_filename,
            )
            item_features += [attr_feat]
            user_features += [user_attr_feat]

        if hp.use_age_info:
            f = PolynomialFeat(FeatureField.USER_AGE, layer_name=FeatureInputLayerName.USER_AGE)
            user_features += [f]

        if hp.market_embed_size > 0:
            n_markets = LabelEncoder.from_model_dir(hp.model_path, LabelEncoderPrefix.MARKET).get_valid_num_items()
            user_market_embed = SparseFeat(
                FeatureField.CLICK_MARKETS,
                vocabulary_size=n_markets + 1,
                embedding_dim=hp.market_embed_size,
                layer_name=FeatureInputLayerName.MARKET_EMBEDDING,
                padding_idx=0,
            )
            item_market_embed = SparseFeat(
                FeatureField.ITEM_MARKET,
                vocabulary_size=n_markets + 1,
                embedding_dim=hp.market_embed_size,
                layer_name=FeatureInputLayerName.MARKET_EMBEDDING,
                padding_idx=0,
            )
            user_features += [user_market_embed]
            item_features += [item_market_embed]

        # complement는 ITEM_CATEGORY 사용 (CBF는 ITEM_STANDARD_CATEGORY)
        if hp.category_embed_size > 0:
            n_categories = LabelEncoder.from_model_dir(
                hp.model_path, LabelEncoderPrefix.CATEGORY
            ).get_valid_num_items()
            user_category_embed = SparseFeat(
                FeatureField.CLICK_CATEGORIES,
                vocabulary_size=n_categories + 1,
                embedding_dim=hp.category_embed_size,
                layer_name=FeatureInputLayerName.CATEGORY_EMBEDDING,
                padding_idx=0,
            )
            item_category_embed = SparseFeat(
                FeatureField.ITEM_CATEGORY,
                vocabulary_size=n_categories + 1,
                embedding_dim=hp.category_embed_size,
                layer_name=FeatureInputLayerName.CATEGORY_EMBEDDING,
                padding_idx=0,
            )
            user_features += [user_category_embed]
            item_features += [item_category_embed]

        user_feat_names = [feat.get_feature_name() for feat in user_features]
        item_feat_names = [feat.get_feature_name() for feat in item_features]
        features = item_features + user_features
        return features, user_feat_names, item_feat_names
