import logging
import os.path
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.constants import (
    EVENT_VOCA,
    ON_VACA_ITEMS_PROCESSED_FILE,
    USER2ITEMS_PROCESSED_FILE,
    DatasetField,
    RawDataField,
)
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    df_user2items: pd.DataFrame
    df_items: pd.DataFrame
    users_static: Dict

    def get_num_vocab_items(self):
        return len(self.df_items)


def _get_raw_user2items_and_on_voca_items(
    load_user2items: Callable,
    user2items_path: Path,
    on_voca_items_path: Path,
    save_preprocessed_data: bool,
) -> Tuple[pd.DataFrame, List]:
    if user2items_path.is_file():
        logger.info("Already user2items df preprocessed. Skip preprocessing")
        df_user2items = pd.read_pickle(user2items_path)
        on_voca_items = pickle.load(on_voca_items_path.open("rb"))
    else:
        if not user2items_path.parent.is_dir():
            user2items_path.parent.mkdir(parents=True)
        df_user2items, on_voca_items = load_user2items()
        logger.info(f"len(on_voca_items): {len(on_voca_items)}")

        if save_preprocessed_data:
            with on_voca_items_path.open("wb") as f:
                pickle.dump(on_voca_items, f)
            df_user2items.to_pickle(user2items_path)
    return df_user2items, on_voca_items


def _create_label_encoders(
    df_items: pd.DataFrame,
    on_voca_items: List,
    model_path: Path,
    use_search_data: bool,
) -> Tuple[LabelEncoder, LabelEncoder, LabelEncoder, LabelEncoder, LabelEncoder]:
    item_encoder = LabelEncoder.from_item_ids(pd.Series(on_voca_items), use_search_data)
    item_encoder.save(model_path, LabelEncoderPrefix.ITEM)

    market_encoder = LabelEncoder.from_item_ids(pd.Series(df_items[RawDataField.MARKET_ID].unique()))
    market_encoder.save(model_path, LabelEncoderPrefix.MARKET)

    category_encoder = LabelEncoder.from_item_ids(pd.Series(df_items[RawDataField.CATEGORY_ID].unique()))
    category_encoder.save(model_path, LabelEncoderPrefix.CATEGORY)

    standard_category_encoder = LabelEncoder.from_item_ids(
        pd.Series(df_items[RawDataField.STANDARD_CATEGORY_ID].unique())
    )
    standard_category_encoder.save(model_path, LabelEncoderPrefix.STANDARD_CATEGORY)

    market_type_encoder = LabelEncoder.from_item_ids(pd.Series(df_items[RawDataField.MARKET_TYPE_ID].unique()))
    market_type_encoder.save(model_path, LabelEncoderPrefix.MARKET_TYPE)
    return item_encoder, market_encoder, category_encoder, standard_category_encoder, market_type_encoder


def _add_index_columns_to_df_items(
    df_items: pd.DataFrame,
    item_encoder: LabelEncoder,
    market_encoder: LabelEncoder,
    category_encoder: LabelEncoder,
    standard_category_encoder: LabelEncoder,
    market_type_encoder: LabelEncoder,
) -> pd.DataFrame:
    df_items[DatasetField.ITEM_INDEX] = df_items[RawDataField.ITEM_ID].astype(str).map(item_encoder.item_id2index)
    df_items[DatasetField.MARKET_INDEX] = df_items[RawDataField.MARKET_ID].astype(str).map(market_encoder.item_id2index)
    df_items[DatasetField.CATEGORY_INDEX] = (
        df_items[RawDataField.CATEGORY_ID].astype(str).map(category_encoder.item_id2index)
    )
    df_items[DatasetField.STANDARD_CATEGORY_INDEX] = (
        df_items[RawDataField.STANDARD_CATEGORY_ID].astype(str).map(standard_category_encoder.item_id2index)
    )
    df_items[DatasetField.MARKET_TYPE_INDEX] = (
        df_items[RawDataField.MARKET_TYPE_ID].astype(str).map(market_type_encoder.item_id2index)
    )
    return df_items


def _add_item_index_column_to_user2items(df_user2items: pd.DataFrame, item_encoder: LabelEncoder) -> pd.DataFrame:
    df_user2query = df_user2items[df_user2items["EVENT_TYPE"] == "query"]
    df_user2items = df_user2items[df_user2items["EVENT_TYPE"] != "query"]

    df_user2items[DatasetField.ITEM_INDEX] = (
        df_user2items[RawDataField.ITEM_ID].astype(str).map(item_encoder.item_id2index).astype(np.int32)
    )

    df_user2query[DatasetField.ITEM_INDEX] = df_user2query[RawDataField.ITEM_ID]
    df_user2items = pd.concat([df_user2items, df_user2query], axis=0)
    return df_user2items


def _add_event_code_column_to_user2items(df_user2items: pd.DataFrame) -> pd.DataFrame:
    df_user2items[DatasetField.EVENT_CODE] = df_user2items[RawDataField.EVENT_TYPE].map(EVENT_VOCA).astype(np.int8)
    return df_user2items


def default_load_raw_df_items(
    goods_filename,
    category_filename,
    standard_category_filename,
    on_voca_items: Optional[List] = None,
    append_category_text: bool = True,
    append_standard_category_text: bool = False,
) -> pd.DataFrame:
    def get_category_map(category_filename):
        df_category = pd.read_csv(category_filename)
        df_category = df_category[["sno", "catnm", "parent_category__sno"]]
        df_category = df_category.set_index("sno")
        sno2text = df_category["catnm"].astype(str).T.squeeze()
        sno2parent = df_category["parent_category__sno"].T.squeeze()
        return sno2text, sno2parent

    def get_standard_category_map(standard_category_filename):
        df_standard_category = pd.read_csv(standard_category_filename)
        df_standard_category = df_standard_category[["sno", "name", "parent_standard_category__sno"]]
        df_standard_category = df_standard_category.set_index("sno")
        sno2text = df_standard_category["name"].astype(str).T.squeeze()
        sno2parent = df_standard_category["parent_standard_category__sno"].T.squeeze()
        return sno2text, sno2parent

    category_sno2text, category_sno2parent = get_category_map(category_filename)
    standard_category_sno2text, standard_category_sno2parent = get_standard_category_map(standard_category_filename)
    df_goods = pd.read_csv(goods_filename, escapechar="\\")
    df_goods = df_goods.drop_duplicates(subset=["sno"])
    df_goods = df_goods.dropna()
    if on_voca_items is not None:
        df_goods = df_goods[df_goods["sno"].isin(on_voca_items)]

    df_goods["category_name"] = df_goods["category_sno"].map(category_sno2text)
    df_goods["category_name"] = df_goods["category_name"].fillna("")
    df_goods["standard_category_name"] = df_goods["standard_category_sno"].map(standard_category_sno2text)
    df_goods["standard_category_name"] = df_goods["standard_category_name"].fillna("")

    if append_category_text and not append_standard_category_text:
        df_goods[DatasetField.ITEM_TEXT] = df_goods["category_name"] + " " + df_goods["name"]
    elif not append_category_text and append_standard_category_text:
        df_goods[DatasetField.ITEM_TEXT] = df_goods["standard_category_name"] + " " + df_goods["name"]
    elif append_category_text and append_standard_category_text:
        df_goods[DatasetField.ITEM_TEXT] = (
            df_goods["category_name"] + " " + df_goods["standard_category_name"] + " " + df_goods["name"]
        )
    else:
        df_goods[DatasetField.ITEM_TEXT] = df_goods["name"]

    df_goods["parent_category__sno"] = df_goods["category_sno"].map(category_sno2parent)
    df_goods['parent_category__sno'] = df_goods['parent_category__sno'].fillna(0)
    df_goods['parent_category__sno'] = df_goods['parent_category__sno'].astype(int)

    df_goods["parent_standard_category__sno"] = df_goods["standard_category_sno"].map(standard_category_sno2parent)
    df_goods['parent_standard_category__sno'] = df_goods['parent_standard_category__sno'].fillna(0)
    df_goods['parent_standard_category__sno'] = df_goods['parent_standard_category__sno'].astype(int)

    df_goods = df_goods[
        [
            "sno",
            DatasetField.ITEM_TEXT,
            RawDataField.MARKET_ID,
            RawDataField.MARKET_TYPE_ID,
            RawDataField.ITEM_PRICE,
            RawDataField.CATEGORY_ID,
            RawDataField.PARENT_CATEGORY_ID,
            RawDataField.STANDARD_CATEGORY_ID,
            RawDataField.PARENT_STANDARD_CATEGORY_ID,
        ]
    ]
    df_goods.columns = [
        RawDataField.ITEM_ID,
        DatasetField.ITEM_TEXT,
        RawDataField.MARKET_ID,
        RawDataField.MARKET_TYPE_ID,
        RawDataField.ITEM_PRICE,
        RawDataField.CATEGORY_ID,
        RawDataField.PARENT_CATEGORY_ID,
        RawDataField.STANDARD_CATEGORY_ID,
        RawDataField.PARENT_STANDARD_CATEGORY_ID,
    ]
    return df_goods


def _load_user_static_info(user_filepath, good_users) -> Dict[str, int]:
    """유저 정보(나이)를 로드. 기존 reco_common preprocess.py와 동일."""
    if not user_filepath.exists():
        return {}

    from utils.df_utils import load_user_df

    df_users = load_user_df(user_filepath)
    df_users = df_users[df_users[RawDataField.USER_ID].isin(good_users)]
    df_users = df_users.set_index(RawDataField.USER_ID)
    dict_users_static = df_users.to_dict()[DatasetField.USER_AGE]
    return dict_users_static


def preprocess(
    load_user2items: Callable,
    load_raw_df_items: Callable,
    preprocessed_root: Path,
    model_path: Path,
    user_file_path: Path,
    use_search_data: bool,
    save_preprocessed_data: bool,
    save_goods_info: bool = False,
):
    """CBF 전처리 파이프라인.

    기존 PytorchLightningFactory.preprocess() + reco_common preprocess()를 통합.
    """
    user2items_path = preprocessed_root.joinpath(USER2ITEMS_PROCESSED_FILE)
    on_voca_items_path = preprocessed_root.joinpath(ON_VACA_ITEMS_PROCESSED_FILE)
    raw_df_user2items, on_voca_items = _get_raw_user2items_and_on_voca_items(
        load_user2items,
        user2items_path,
        on_voca_items_path,
        save_preprocessed_data,
    )
    raw_df_items: pd.DataFrame = load_raw_df_items(on_voca_items)

    (
        item_encoder,
        market_encoder,
        category_encoder,
        standard_category_encoder,
        market_type_encoder,
    ) = _create_label_encoders(
        raw_df_items,
        on_voca_items,
        model_path,
        use_search_data,
    )

    df_user2items: pd.DataFrame = _add_item_index_column_to_user2items(raw_df_user2items, item_encoder)
    df_user2items: pd.DataFrame = _add_event_code_column_to_user2items(df_user2items)
    df_user2items = df_user2items[
        [
            RawDataField.USER_ID,
            DatasetField.ITEM_INDEX,
            RawDataField.TIMESTAMP,
            DatasetField.EVENT_CODE,
        ]
    ]

    df_items: pd.DataFrame = _add_index_columns_to_df_items(
        raw_df_items,
        item_encoder,
        market_encoder,
        category_encoder,
        standard_category_encoder,
        market_type_encoder,
    )

    if save_goods_info:
        df_items.to_pickle(os.path.join(model_path, "df_goods.pkl"))

    df_items = df_items[
        [
            DatasetField.ITEM_INDEX,
            DatasetField.ITEM_TEXT,
            DatasetField.MARKET_INDEX,
            DatasetField.MARKET_TYPE_INDEX,
            DatasetField.ITEM_PRICE,
            DatasetField.CATEGORY_INDEX,
            DatasetField.STANDARD_CATEGORY_INDEX,
            RawDataField.ITEM_ID,
        ]
    ]
    logger.info(f"len(df_items): {len(df_items)}")

    users_static = _load_user_static_info(user_file_path, good_users=list(df_user2items[RawDataField.USER_ID]))
    return PreprocessResult(
        df_user2items=df_user2items,
        df_items=df_items,
        users_static=users_static,
    )
