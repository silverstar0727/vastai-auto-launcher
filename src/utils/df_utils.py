"""기존 reco_common/util/data_utils/df_utils.py 원본 그대로 포팅."""
import logging
import os
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from utils.constants import EVENT_MAP, DatasetField, RawDataField

logger = logging.getLogger(__name__)


def read_table(path, **csv_kwargs):
    """확장자 보고 parquet/csv 자동 분기. 디렉터리는 parquet으로 간주."""
    p = str(path)
    if p.endswith(".parquet") or os.path.isdir(p):
        return pd.read_parquet(p)
    return pd.read_csv(p, **csv_kwargs)


def load_df_user2items(interaction_csv, order_csv, event_map=EVENT_MAP):
    # USER_ID  ITEM_ID   TIMESTAMP EVENT_TYPE  EVENT_VALUE
    df_interaction = _load_interaction_df(interaction_csv, event_map=event_map)

    # USER_ID  ITEM_ID   TIMESTAMP EVENT_TYPE
    df_order = _load_order_df(order_csv)
    df_order[RawDataField.EVENT_TYPE] = event_map["order"]

    df = pd.concat([df_interaction, df_order], axis=0, ignore_index=True)

    num_preferences = len(df[df[RawDataField.EVENT_TYPE] == "preference"])
    num_clicks = len(df[df[RawDataField.EVENT_TYPE] == "click"])
    logger.info(f"clicks: {num_clicks}, preferences: {num_preferences}, ")
    return df


def _load_order_df(csv_filepath):
    df = read_table(csv_filepath)
    df = df[["order_sno", "goods_sno", "member_sno"]]
    df = df[df["member_sno"].notna()].astype({"member_sno": "int32"})
    df[RawDataField.USER_ID] = df["member_sno"].astype(str).map("m{}".format)
    df[RawDataField.TIMESTAMP] = (df["order_sno"] / 1000).astype(int)

    df = df[[RawDataField.USER_ID, "goods_sno", RawDataField.TIMESTAMP]]
    df.columns = [RawDataField.USER_ID, RawDataField.ITEM_ID, RawDataField.TIMESTAMP]
    return df


def _load_interaction_df(csv_filepath, event_map=EVENT_MAP):
    files = glob(str(csv_filepath) + "/*.csv")
    df_list = (pd.read_csv(file) for file in files)
    df = pd.concat(df_list, ignore_index=True)
    # ITEM_ID 에 nan이 섞여있는 경우가 있음.
    df = df.dropna()
    df = df.astype({"ITEM_ID": int})

    num_likes = len(df[df[RawDataField.EVENT_TYPE] == "like"])
    num_carts = len(df[df[RawDataField.EVENT_TYPE] == "cart"])
    num_deeplink = len(df[df[RawDataField.EVENT_TYPE] == "deeplink"])
    logger.info(f"likes: {num_likes}, carts: {num_carts}, deeplink: {num_deeplink}")

    df[RawDataField.EVENT_TYPE] = df[RawDataField.EVENT_TYPE].map(event_map)

    df = df.dropna()
    df = df.astype({RawDataField.ITEM_ID: "int32"})
    df = df[
        [
            RawDataField.USER_ID,
            RawDataField.ITEM_ID,
            RawDataField.TIMESTAMP,
            RawDataField.EVENT_TYPE,
        ]
    ]
    return df


def load_user_df(csv_filepath, valid_min_age=10, valid_max_age=40, invalid_birth="1990-01-01"):
    def diff_to_age(diff):
        return int(diff / np.timedelta64(365, "D"))

    df = read_table(csv_filepath)
    df = df[df["birth_date"].notna()]

    df[RawDataField.USER_ID] = df["m_no"].astype(str).map("m{}".format)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df = df[~df["birth_date"].isnull()]
    if invalid_birth is not None:
        df = df[~(df["birth_date"] == invalid_birth)]

    df[DatasetField.USER_AGE] = pd.Timestamp("now") - df["birth_date"]
    df[DatasetField.USER_AGE] = df[DatasetField.USER_AGE].apply(diff_to_age)
    df = df[[RawDataField.USER_ID, DatasetField.USER_AGE]]

    df = df[df[DatasetField.USER_AGE] >= valid_min_age]
    df = df[df[DatasetField.USER_AGE] <= valid_max_age]
    return df


def filter_popular_items(df: pd.DataFrame, max_items: Optional[int]) -> List:
    logger.info("Filtering triplets")
    logger.info("Total number of items : {}".format(len(df.groupby(RawDataField.ITEM_ID).size())))

    n_actions_per_item: pd.DataFrame = df.groupby(RawDataField.ITEM_ID).size()
    n_actions_per_item = n_actions_per_item.to_frame(name="n_actions")

    n_actions_per_item = n_actions_per_item.sort_values(by=["n_actions", RawDataField.ITEM_ID], ascending=False)

    if max_items is not None and len(n_actions_per_item) > max_items:
        logger.info(f"Filter items below {n_actions_per_item.iloc[max_items]} interactions")
        n_actions_per_item = n_actions_per_item[:max_items]

    popular_items = list(n_actions_per_item.index)
    logger.info("Popularity filtered number of items : {}".format(len(popular_items)))
    return popular_items


def filter_items_by_min_actions(df, min_actions_per_item):
    logger.info("Filtering triplets")
    logger.info("Total number of items : {}".format(len(df.groupby(RawDataField.ITEM_ID).size())))
    n_actions_per_item: pd.Series = df.groupby(RawDataField.ITEM_ID).size()
    n_actions_per_item = n_actions_per_item.sort_values(ascending=False)
    return list(n_actions_per_item[n_actions_per_item >= min_actions_per_item].index)


def filter_short_seq(df, min_actions_per_user, count_unique=False):
    if min_actions_per_user > 0:
        if count_unique:
            n_actions_per_user: pd.Series = df.groupby(RawDataField.USER_ID)[RawDataField.ITEM_ID].nunique()
        else:
            n_actions_per_user: pd.Series = df.groupby(RawDataField.USER_ID).size()
        good_users = n_actions_per_user.index[n_actions_per_user >= min_actions_per_user]
        logger.info(f"len(good_users): {len(good_users)}")
        df = df[df[RawDataField.USER_ID].isin(good_users)]
    return df


def find_items_by_preference_per_click(
    df_input: pd.DataFrame,
    preference_per_click_condition: Dict,
) -> List[int]:
    try:
        df_item_action_stat: pd.DataFrame = (
            df_input.groupby([RawDataField.ITEM_ID, RawDataField.EVENT_TYPE])
            .agg(cnt=(RawDataField.USER_ID, len))
            .reset_index()
            .pivot(index=RawDataField.ITEM_ID, columns=RawDataField.EVENT_TYPE, values='cnt')
            .fillna({'click': 0.0, 'preference': 0.0})
            .reset_index()
            .assign(preference_per_click=lambda x: x['preference'] / (1.0 + x['click']))[
                [RawDataField.ITEM_ID, 'click', 'preference_per_click']
            ]
        )
    except Exception:
        return []

    found_items: List[int] = list(
        df_item_action_stat.query(f'click >= {preference_per_click_condition["min_click_threshold"]}').query(
            f'preference_per_click < {preference_per_click_condition["preference_per_click_threshold"]}'
        )[RawDataField.ITEM_ID]
    )
    return found_items
