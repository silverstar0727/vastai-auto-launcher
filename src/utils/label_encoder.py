import json
import logging
import os

import numpy as np
import pandas as pd
from utils.constants import (
    MASK_ITEM_ID,
    QUERY_ITEM_ID,
    SPECIAL_TOKENS,
    UNKNOWN_ITEM_ID,
    UNKNOWN_ITEM_INDEX,
)

logger = logging.getLogger(__name__)


class LabelEncoderPrefix:
    ITEM = "item"
    MARKET = "market"
    CATEGORY = "category"
    STANDARD_CATEGORY = "standard_category"
    MARKET_TYPE = "market_type"
    TASTE = "taste"
    OUTPUT_ITEM = "output_item"
    USER = "user"
    PRODUCT = "product"
    PARENT_CATEGORY = "parent_category"


def _load_item_meta(model_dir, prefix):
    """Load index2id and id2index JSON files from model directory."""
    index2id_path = os.path.join(model_dir, f"{prefix}_index2id.json")
    id2index_path = os.path.join(model_dir, f"{prefix}_id2index.json")
    with open(index2id_path, "r") as f:
        item_index2id = json.load(f)
    with open(id2index_path, "r") as f:
        item_id2index = json.load(f)
    return item_index2id, item_id2index


class LabelEncoder(object):
    def __init__(self, item_index2id, item_id2index, id_dtype=int):
        self.item_index2id = item_index2id
        self.item_id2index = item_id2index
        self.id_dtype = id_dtype

    @classmethod
    def from_model_dir(cls, model_dir, prefix, id_type=int):
        try:
            item_index2id, item_id2index = _load_item_meta(model_dir, prefix)
            return LabelEncoder(item_index2id, item_id2index, id_type)
        except Exception:
            return None

    @classmethod
    def from_item_ids(cls, item_ids: pd.Series, use_search_token=False, id_dtype=int):
        item_index2id, item_id2index = _create_item_map(item_ids, use_search_token, id_dtype=id_dtype)
        return LabelEncoder(item_index2id, item_id2index, id_dtype)

    @classmethod
    def unknown_item_id(cls):
        return UNKNOWN_ITEM_ID

    @classmethod
    def unknown_item_index(cls):
        return UNKNOWN_ITEM_INDEX

    @classmethod
    def mask_item_id(cls):
        return MASK_ITEM_ID

    def mask_item_index(self):
        return self.item_id2index[MASK_ITEM_ID]

    def to_ids(self, item_indexes) -> list:
        item_ids = [self.item_index2id[str(item)] for item in item_indexes]
        return item_ids

    def to_indexes(self, item_ids):
        items = []
        oov_item_ids = []
        for item in item_ids:
            item_idx = self.item_id2index.get(str(item), UNKNOWN_ITEM_INDEX)
            if item_idx == UNKNOWN_ITEM_INDEX:
                oov_item_ids.append(self.id_dtype(item))
            else:
                items.append(item_idx)
        return items, oov_item_ids

    def to_indexes_reserving_size(self, item_ids):
        return [self.item_id2index.get(str(item), UNKNOWN_ITEM_INDEX) for item in item_ids]

    def save(self, model_path, prefix=LabelEncoderPrefix.ITEM):
        with open(os.path.join(model_path, f"{prefix}_index2id.json"), "w") as f:
            json.dump(self.item_index2id, f)
        with open(os.path.join(model_path, f"{prefix}_id2index.json"), "w") as f:
            json.dump(self.item_id2index, f)
        logger.info(f"{prefix}_index2id.json created")
        logger.info(f"{prefix}_id2index.json created")

    def get_valid_num_items(self):
        return len(self.get_valid_item_ids())

    def get_valid_item_ids(self):
        item_ids = list(self.item_id2index.keys())
        for t in SPECIAL_TOKENS:
            if t in item_ids:
                item_ids.remove(t)
        item_ids = np.array(item_ids, dtype=self.id_dtype)
        return item_ids

    def extract_valid_actions(self, item_ids, events, time_steps):
        bool_mask = self._get_valid_item_bool_mask(item_ids)
        item_ids = np.array(item_ids)[bool_mask].tolist()
        events = np.array(events)[bool_mask].tolist()
        time_steps = np.array(time_steps)[bool_mask].tolist()
        return item_ids, events, time_steps

    def _get_valid_item_bool_mask(self, item_ids):
        bool_mask = np.zeros((len(item_ids)), dtype=bool)
        for i, item in enumerate(item_ids):
            if str(item) in self.item_id2index.keys():
                bool_mask[i] = True
        return bool_mask


def _create_item_map(ids: pd.Series, use_search_token, id_dtype: type = int):
    item_id2index = {
        UNKNOWN_ITEM_ID: UNKNOWN_ITEM_INDEX,
        MASK_ITEM_ID: len(list(set(ids))) + 1,
    }
    if use_search_token:
        item_id2index[QUERY_ITEM_ID] = item_id2index[MASK_ITEM_ID] + 2

    for i, item_id in enumerate(ids.unique()):
        item_id2index[str(item_id)] = i + 1

    item_index2id = {}
    for id_, index in item_id2index.items():
        if id_ == UNKNOWN_ITEM_ID or id_ == MASK_ITEM_ID or id_ == QUERY_ITEM_ID:
            item_index2id[str(index)] = id_
        else:
            item_index2id[str(index)] = id_dtype(id_)
    return item_index2id, item_id2index
