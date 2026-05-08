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
from utils.constants import DatasetField, FeatureField, RawDataField
from utils.data_handler import DatasetHandler, split_data_by_users
from utils.df_utils import read_table
from utils.goods_info import GoodsInfo
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix
from utils.preprocess import PreprocessResult, default_load_raw_df_items
from utils.price import add_item_price_group_column, get_standard_category_price_info

logger = logging.getLogger(__name__)


def _get_on_sale_items(goods_filename):
    """판매중인 아이템 목록 반환. DefaultRawDfLoader._get_on_sale_items()와 동일."""
    df_items = read_table(goods_filename, escapechar="\\")
    df_items = df_items.dropna()
    return list(df_items["sno"])


# --- Dataset Classes (원본 dataloaders.py 그대로) ---


def _insert_pad(seq, max_len):
    pad_len = max_len - len(seq)
    seq_padded = [0] * pad_len + seq
    seq_padded = seq_padded[-max_len:]
    return seq_padded


class CBFTrainDataset(data_utils.Dataset):
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


class CBFInBatchNegProbTrainDataset(data_utils.Dataset):
    def __init__(
        self,
        dataset_handler: DatasetHandler,
        max_len: int,
        num_items: int,
        rng: random.Random,
        max_pos_items: int = 400,
    ):
        self.dataset_handler = dataset_handler
        self.max_len = max_len
        self.num_items = num_items
        self.rng = rng
        self.max_pos_items = max_pos_items
        self.sample_probs = self._get_sample_probs()

    def __len__(self):
        return self.dataset_handler.num_train_samples()

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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

        if len(output_items) > self.max_pos_items:
            pos_labels = self.rng.sample(output_items.tolist(), self.max_pos_items)
        else:
            pos_labels = _insert_pad(output_items.tolist(), self.max_pos_items)

        pos_pops = [min(self.sample_probs[x], 1e-7) for x in pos_labels]

        return input_tensors, torch.LongTensor(pos_labels), torch.FloatTensor(pos_pops)

    def _get_sample_probs(self):
        item_freq_info = self.dataset_handler.get_item_frequency_info("train")
        probs = np.zeros(self.num_items + 1)
        items = np.array(list(item_freq_info.keys()))
        freqs = np.array(list(item_freq_info.values()))
        probs[items] = freqs
        probs = probs / probs.sum()
        return probs


class CBFEvalDataset(data_utils.Dataset):
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


class CBFDataModule(L.LightningDataModule):
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
        order_filename: str = "",
        # 데이터 필터링
        min_actions_per_user: int = 50,
        max_items: int = 1000000,
        # 배치 설정
        batch_size: int = 1024,
        test_batch_size: int = 1024,
        max_len: int = 200,
        num_workers: int = 0,
        drop_last_batch: bool = True,
        # 피처 설정
        item_embed_size: int = 256,
        item_vocabulary_size: int = 120000,
        use_age_info: bool = True,
        use_in_house_text_embed: bool = True,
        use_item_emb_in_item_tower: bool = False,
        use_attr: bool = False,
        attr_meta_filename: str = "",
        attr_filename: str = "",
        market_embed_size: int = 128,
        category_embed_size: int = 128,
        price_group_embed_size: int = 0,
        # 학습 설정
        negative_sampling: bool = True,
        bias_correction: bool = False,
        # 데이터 분할
        max_test_samples: int = 200000,
        num_test_samples_per_user: int = 10,
        # 시드
        dataloader_random_seed: float = 0.0,
        # preference/click 필터
        preference_per_click_condition: Optional[Dict] = None,
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

        # 4. item side features
        self.item_feat_values = self._get_item_side_feature(self.goods_info)

        # 5. num_items (모델의 embedding vocab = goods_info items + 1 for padding)
        self.num_items = self.goods_info.get_valid_num_items() + 1

        # 6. 유저 분할
        model_feature_names = list(set(user_feat_names + item_feat_names))
        dataset_handler = self._load_dataset_handler(
            preprocess_result, model_feature_names
        )

        # 7. Dataset 생성
        # 데이터셋의 num_items는 전처리된 전체 아이템 수 (multi-hot label 크기 결정)
        dataset_num_items = preprocess_result.get_num_vocab_items()
        rng = random.Random(hp.dataloader_random_seed)

        if hp.bias_correction:
            assert hp.negative_sampling, "Bias correction is only available with negative sampling"
            self._train_dataset = CBFInBatchNegProbTrainDataset(
                dataset_handler, hp.max_len, dataset_num_items, rng,
            )
        else:
            self._train_dataset = CBFTrainDataset(
                dataset_handler, hp.max_len, dataset_num_items, rng,
                use_negative_sampling=hp.negative_sampling,
            )
        self._eval_dataset = CBFEvalDataset(dataset_handler, hp.max_len)

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

    # --- Private methods (기존 pl_factory.py + basic_pl.py 로직) ---

    def _run_preprocess(self) -> PreprocessResult:
        """기존 PytorchLightningFactory.preprocess() + DefaultRawDfLoader 로직."""
        hp = self.hparams
        raw_dataset_root = Path(hp.raw_dataset_root)
        model_path = Path(hp.model_path)
        preprocessed_root = model_path.joinpath("preprocessed")

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(preprocessed_root, exist_ok=True)

        goods_filename = hp.goods_filename
        category_filename = hp.category_filename
        standard_category_filename = hp.standard_category_filename

        def load_user2items():
            """DefaultRawDfLoader.load_raw_user2items() 완전 동일 구현."""
            return self._load_raw_user2items()

        def load_raw_df_items(on_voca_items):
            """DefaultRawDfLoader.load_raw_df_items() 완전 동일 구현."""
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
        """DefaultRawDfLoader.load_raw_user2items()와 완전히 동일한 구현."""
        hp = self.hparams
        from utils.df_utils import (
            filter_popular_items,
            filter_short_seq,
            find_items_by_preference_per_click,
            load_df_user2items,
        )

        interaction_filepath = Path(hp.raw_dataset_root)
        order_filepath = Path(hp.order_filename) if hp.order_filename else Path(hp.sql_dataset_root) / "order.csv"
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
        logger.info(f"2. min user filter :  # of interactions {len(df)}")

        # 2.5 preference/click 비율 필터
        pref_per_click_deny_items = find_items_by_preference_per_click(
            df, preference_per_click_condition,
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
        logger.info(f"4. min user filter :  # of interactions {len(df)}")

        return df, on_voca_items

    def _load_dataset_handler(self, preprocess_result: PreprocessResult, feature_names: List[str]) -> DatasetHandler:
        """기존 basic_pl.py의 _load_dataset_handler() 로직."""
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
        """기존 CbfPytorchLightningFactory._get_feature_process_spec() 그대로."""
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

        if hp.category_embed_size > 0:
            n_categories = LabelEncoder.from_model_dir(
                hp.model_path, LabelEncoderPrefix.STANDARD_CATEGORY
            ).get_valid_num_items()
            user_category_embed = SparseFeat(
                FeatureField.CLICK_CATEGORIES,
                vocabulary_size=n_categories + 1,
                embedding_dim=hp.category_embed_size,
                layer_name=FeatureInputLayerName.CATEGORY_EMBEDDING,
                padding_idx=0,
            )
            item_category_embed = SparseFeat(
                FeatureField.ITEM_STANDARD_CATEGORY,
                vocabulary_size=n_categories + 1,
                embedding_dim=hp.category_embed_size,
                layer_name=FeatureInputLayerName.CATEGORY_EMBEDDING,
                padding_idx=0,
            )
            user_features += [user_category_embed]
            item_features += [item_category_embed]

        if hp.price_group_embed_size > 0:
            n_price_groups = 3
            user_price_group_embed = SparseFeat(
                FeatureField.CLICK_PRICE_GROUPS,
                vocabulary_size=n_price_groups + 1,
                embedding_dim=hp.price_group_embed_size,
                layer_name=FeatureInputLayerName.PRICE_GROUP_EMBEDDING,
                padding_idx=0,
            )
            item_price_group_embed = SparseFeat(
                FeatureField.ITEM_PRICE_GROUP,
                vocabulary_size=n_price_groups + 1,
                embedding_dim=hp.price_group_embed_size,
                layer_name=FeatureInputLayerName.PRICE_GROUP_EMBEDDING,
                padding_idx=0,
            )
            user_features += [user_price_group_embed]
            item_features += [item_price_group_embed]

        user_feat_names = [feat.get_feature_name() for feat in user_features]
        item_feat_names = [feat.get_feature_name() for feat in item_features]
        features = item_features + user_features
        return features, user_feat_names, item_feat_names

    def _get_item_side_feature(self, goods_info: GoodsInfo) -> Dict:
        """기존 CbfPytorchLightningFactory._get_item_side_feature() 그대로."""
        hp = self.hparams
        feature_fields = [
            FeatureField.ITEM_MARKET,
            FeatureField.ITEM_STANDARD_CATEGORY,
        ]
        if hp.price_group_embed_size > 0:
            feature_fields.append(FeatureField.ITEM_PRICE)

        item_side_features = goods_info.get_item_side_features_selectively(feature_fields)

        if hp.price_group_embed_size > 0:
            price_group_features = self._compute_item_price_group(item_side_features)
            item_side_features.update(price_group_features)

        return item_side_features

    def _compute_item_price_group(self, item_side_features: Dict) -> Dict:
        """기존 CbfPytorchLightningFactory._compute_item_price_group() 그대로."""
        hp = self.hparams
        df_items = pd.DataFrame({k: v.reshape(-1) for k, v in item_side_features.items()})

        df_standard_category_price_percentile = get_standard_category_price_info(df_items)
        df_standard_category_price_percentile.to_pickle(
            os.path.join(hp.model_path, 'standard_category_price_percentile.pkl')
        )

        df_items_price_group = add_item_price_group_column(df_items, df_standard_category_price_percentile)[
            [FeatureField.ITEM_PRICE_GROUP]
        ]
        price_group_features = {
            FeatureField.ITEM_PRICE_GROUP: df_items_price_group.values,
        }
        return price_group_features
