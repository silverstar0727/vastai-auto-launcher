import logging
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from utils.constants import DatasetField, FeatureField, RawDataField

logger = logging.getLogger(__name__)


class DatasetHandler(object):
    def __init__(
        self,
        train_user2items: list,
        test_user2items: list,
        user_ages: np.array,
        user_codes: list,
        feature_names: list,
    ):
        self.train_user2items = train_user2items
        self.test_user2items = test_user2items
        self.user_ages = user_ages
        self.user_codes = user_codes
        self.feature_names = feature_names

    @classmethod
    def from_pickle(cls, dataset_path):
        dataset = pickle.load(dataset_path.open("rb"))
        dataset["user_ages"] = np.array(dataset["user_ages"], dtype=np.uint8)
        return DatasetHandler(**dataset)

    def save(self, dataset_path):
        dataset = {
            "train_user2items": self.train_user2items,
            "test_user2items": self.test_user2items,
            "user_ages": self.user_ages,
            "user_codes": self.user_codes,
            "feature_names": self.feature_names,
        }
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    def num_train_samples(self):
        return len(self.train_user2items)

    def num_test_samples(self):
        return len(self.test_user2items)

    def get_user2items(self, index, data="train"):
        if data == "train":
            df = self.train_user2items[index]
        else:
            df = self.test_user2items[index]
        return df

    def get_items(self, index, data="train") -> list:
        df = self.get_user2items(index, data)
        items = list(df[DatasetField.ITEM_INDEX])
        return items

    def get_user_info(self, index) -> Dict[str, torch.Tensor]:
        user_info = {
            FeatureField.USER_CODE: self.user_codes[index],
        }
        if FeatureField.use_age_info(self.feature_names):
            age = self.user_ages[index] / 50
            user_info[FeatureField.USER_AGE] = torch.FloatTensor([age])
        return user_info

    def get_item_frequency_info(self, data="train") -> Dict[int, int]:
        if data == "train":
            user2items = self.train_user2items
        else:
            user2items = self.test_user2items
        item_freq = {}
        for df in user2items:
            for item in df[DatasetField.ITEM_INDEX]:
                item_freq[item] = item_freq.get(item, 0) + 1
        return item_freq


def split_data_by_users(
    df_user2items,
    dict_users_static,
    feature_names,
    max_test_samples: Optional[int] = 200000,
    max_seq_per_user: Optional[int] = None,
    seq_len: Optional[int] = None,
    random_seed: Optional[str] = "111",
    n_test_samples_per_seq: int = 1,
    remove_duplicate_in_seq: bool = False,
    remove_duplicate_by_event_prior: bool = True,
) -> DatasetHandler:
    rng = _get_rng(random_seed)

    train_user2items = []
    test_user2items = []

    user_grouped = df_user2items.groupby(RawDataField.USER_ID)
    user_ages = []
    user_codes: list = []
    n_user2items_train = 0
    n_user2items_test = 0

    for i, (user_id, df_user2items_per_user) in tqdm(enumerate(user_grouped)):
        del df_user2items_per_user[RawDataField.USER_ID]
        if remove_duplicate_in_seq:
            df_user2items_per_user = _remove_duplicate_items(df_user2items_per_user, remove_duplicate_by_event_prior)

        df_user2items_per_user = df_user2items_per_user.sort_values(by=RawDataField.TIMESTAMP)
        if max_seq_per_user is not None:
            df_user2items_per_user = df_user2items_per_user[-max_seq_per_user:]

        n_samples = _get_num_splits(df_user2items_per_user, seq_len)
        batch_dfs = np.array_split(df_user2items_per_user, n_samples)

        for df in batch_dfs:
            df = df.reset_index(drop=True)

            if FeatureField.use_time_info(feature_names):
                df = _timestamp_to_interval(df)
            else:
                df = df.drop(columns=[RawDataField.TIMESTAMP])

            user_codes.append(user_id)
            if FeatureField.use_age_info(feature_names):
                user_ages.append(dict_users_static.get(user_id, 0))

            n_test_samples = n_test_samples_per_seq if len(test_user2items) < max_test_samples else 0
            train_indexes, test_indexes = _get_item_indexes(
                df[DatasetField.ITEM_INDEX],
                rng=rng,
                n_test_samples_per_seq=n_test_samples,
            )

            train_user2items.append(df.iloc[train_indexes])
            if test_indexes is not None:
                test_user2items.append(df.iloc[test_indexes])
                n_user2items_test += len(test_indexes)
            n_user2items_train += len(train_indexes)

    logger.info(
        f"{len(user_grouped)}-Users, {n_user2items_train}-User2Items(Train), {n_user2items_test}-User2Items(Test)"
    )
    return DatasetHandler(
        train_user2items,
        test_user2items,
        np.array(user_ages, dtype=np.uint8),
        user_codes,
        feature_names,
    )


def _get_rng(random_seed):
    if random_seed is None:
        rng = None
    else:
        rng = random.Random(random_seed)
    return rng


def _get_num_splits(items, seq_len):
    if seq_len is not None:
        n_samples = max(len(items) // (seq_len + 1), 1)
    else:
        n_samples = 1
    return n_samples


def _get_item_indexes(items, rng, n_test_samples_per_seq):
    indexes = np.arange(len(items))

    if n_test_samples_per_seq == 0:
        test_indexes = None
    else:
        if rng is None:
            test_indexes = indexes[-n_test_samples_per_seq:]
        else:
            # 첫 번째 item은 test sample로 사용 안함.
            candidate_indexes = indexes.tolist()[1:]
            if len(candidate_indexes) < n_test_samples_per_seq:
                test_indexes = candidate_indexes
            else:
                test_indexes = rng.sample(candidate_indexes, n_test_samples_per_seq)

    train_indexes = np.setdiff1d(indexes, test_indexes)
    return train_indexes, test_indexes


def _remove_duplicate_items(df, remove_duplicate_by_event_prior):
    if remove_duplicate_by_event_prior:
        df = df.sort_values(by=DatasetField.EVENT_CODE, ascending=False)
        df = df.drop_duplicates(subset=[DatasetField.ITEM_INDEX], keep="first")
        df = df.sort_values(by=RawDataField.TIMESTAMP)
    else:
        df = df.drop_duplicates(subset=[DatasetField.ITEM_INDEX], keep="first")
    return df


def _timestamp_to_interval(df):
    times = df[RawDataField.TIMESTAMP].values
    time_intervals = times[1:] - times[:-1]
    ONE_WEEK_AS_SECOND = 604800
    normalized = []
    for interval in time_intervals:
        val = interval / ONE_WEEK_AS_SECOND
        val = min(val, 1.0)
        normalized.append(val)
    df[RawDataField.TIMESTAMP] = np.array([0] + normalized, dtype=np.float32)
    df.columns = [
        DatasetField.ITEM_INDEX,
        DatasetField.TIME_INTERVAL,
        DatasetField.EVENT_CODE,
    ]
    return df


def split_train_test_users(
    all_users: List[str], random_seed: int = 111, test_user_ratio: float = 0.05
) -> Tuple[List[str], List[str]]:
    rng = random.Random(random_seed)
    num_test_users = max(1, int(len(all_users) * test_user_ratio))
    test_users = rng.sample(all_users, num_test_users)
    train_users = np.setdiff1d(all_users, test_users).tolist()
    return train_users, test_users
