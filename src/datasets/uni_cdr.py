import logging
import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

import lightning as L

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User type constants
# ---------------------------------------------------------------------------

USER_COMMON = 0  # user exists in both domains
USER_SOURCE = 1  # user exists only in source domain
USER_TARGET = 2  # user exists only in target domain

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class UniCDRTrainDataset(data_utils.Dataset):
    """Training dataset with random 20% dropout on click history."""

    def __init__(
        self,
        samples: list[dict],
        src_num_items: int,
        target_num_items: int,
        src_max_len: int,
        target_max_len: int,
        dropout_rate: float = 0.2,
        rng: random.Random | None = None,
    ):
        self.samples = samples
        self.src_num_items = src_num_items
        self.target_num_items = target_num_items
        self.src_max_len = src_max_len
        self.target_max_len = target_max_len
        self.dropout_rate = dropout_rate
        self.rng = rng or random.Random(42)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        src_inputs = self._build_sequence(
            sample["src_item_ids"],
            sample["src_market_ids"],
            sample["src_category_ids"],
            self.src_max_len,
            apply_dropout=True,
        )
        target_inputs = self._build_sequence(
            sample["target_item_ids"],
            sample["target_market_ids"],
            sample["target_category_ids"],
            self.target_max_len,
            apply_dropout=True,
        )

        src_inputs["n_interactions"] = sample["n_src"]
        target_inputs["n_interactions"] = sample["n_target"]
        src_inputs["cand_item_ids"] = torch.LongTensor(sample["src_cand_items"])
        target_inputs["cand_item_ids"] = torch.LongTensor(sample["target_cand_items"])

        user_type = sample["user_type"]
        src_label = sample["src_label"]
        target_label = sample["target_label"]

        return (
            src_inputs,
            target_inputs,
            user_type,
            src_label,
            target_label,
        )

    def _build_sequence(self, item_ids, market_ids, category_ids, max_len, apply_dropout=False):
        items = list(item_ids)
        markets = list(market_ids)
        categories = list(category_ids)

        # Random 20% dropout on history
        if apply_dropout and len(items) > 1:
            keep = []
            for i in range(len(items)):
                if self.rng.random() > self.dropout_rate:
                    keep.append(i)
            if len(keep) == 0:
                keep = [len(items) - 1]
            items = [items[i] for i in keep]
            markets = [markets[i] for i in keep]
            categories = [categories[i] for i in keep]

        # Truncate and pad (left-pad)
        items = items[-max_len:]
        markets = markets[-max_len:]
        categories = categories[-max_len:]

        pad_len = max_len - len(items)
        items = [0] * pad_len + items
        markets = [0] * pad_len + markets
        categories = [0] * pad_len + categories

        return {
            "item_ids": torch.LongTensor(items),
            "market_ids": torch.LongTensor(markets),
            "category_ids": torch.LongTensor(categories),
        }


class UniCDREvalDataset(data_utils.Dataset):
    """Evaluation dataset for target-domain prediction."""

    def __init__(
        self,
        samples: list[dict],
        src_max_len: int,
        target_max_len: int,
    ):
        self.samples = samples
        self.src_max_len = src_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        src_inputs = self._build_sequence(
            sample["src_item_ids"],
            sample["src_market_ids"],
            sample["src_category_ids"],
            self.src_max_len,
        )
        target_inputs = self._build_sequence(
            sample["target_item_ids"],
            sample["target_market_ids"],
            sample["target_category_ids"],
            self.target_max_len,
        )

        src_inputs["n_interactions"] = sample["n_src"]
        target_inputs["n_interactions"] = sample["n_target"]
        src_inputs["cand_item_ids"] = torch.LongTensor(sample["src_cand_items"])
        target_inputs["cand_item_ids"] = torch.LongTensor(sample["target_cand_items"])

        user_type = sample["user_type"]
        src_label = sample["src_label"]
        target_label = sample["target_label"]

        return (
            src_inputs,
            target_inputs,
            user_type,
            src_label,
            target_label,
        )

    def _build_sequence(self, item_ids, market_ids, category_ids, max_len):
        items = list(item_ids)
        markets = list(market_ids)
        categories = list(category_ids)

        items = items[-max_len:]
        markets = markets[-max_len:]
        categories = categories[-max_len:]

        pad_len = max_len - len(items)
        items = [0] * pad_len + items
        markets = [0] * pad_len + markets
        categories = [0] * pad_len + categories

        return {
            "item_ids": torch.LongTensor(items),
            "market_ids": torch.LongTensor(markets),
            "category_ids": torch.LongTensor(categories),
        }


# ---------------------------------------------------------------------------
# Collate function for nested dict batches
# ---------------------------------------------------------------------------


def _collate_uni_cdr(batch):
    """Custom collate for UniCDR samples with nested dict inputs."""
    src_inputs_list, target_inputs_list, user_types, src_labels, target_labels = zip(*batch)

    def stack_dict(dict_list):
        result = {}
        for key in dict_list[0]:
            vals = [d[key] for d in dict_list]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)
            else:
                result[key] = torch.tensor(vals)
        return result

    return (
        stack_dict(src_inputs_list),
        stack_dict(target_inputs_list),
        torch.tensor(user_types, dtype=torch.long),
        torch.tensor(src_labels, dtype=torch.long),
        torch.tensor(target_labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class UniCDRDataModule(L.LightningDataModule):
    """UniCDR DataModule.

    Loads parquet interaction data, splits by domain category (source/target),
    classifies users as COMMON/SOURCE/TARGET, and builds train/eval datasets.
    """

    def __init__(
        self,
        interaction_dir: str,
        goods_path: str = "",
        pretrained_text_path: str = "",
        cache_dir: str = "",
        src_category_sno: int = 1,
        target_category_sno: int = 378,
        src_max_items: int = 100000,
        target_max_items: int = 100000,
        seq_max_len: int = 200,
        transformer_input_len: int = 50,
        src_sampling_ratio: float = 0.05,
        target_sampling_ratio: float = 0.2,
        query_pad_length: int = 5,
        batch_size: int = 640,
        test_batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.interaction_dir = interaction_dir
        self.goods_path = goods_path
        self.pretrained_text_path = pretrained_text_path
        self.cache_dir = cache_dir
        self.src_category_sno = src_category_sno
        self.target_category_sno = target_category_sno
        self.src_max_items = src_max_items
        self.target_max_items = target_max_items
        self.seq_max_len = seq_max_len
        self.transformer_input_len = transformer_input_len
        self.src_sampling_ratio = src_sampling_ratio
        self.target_sampling_ratio = target_sampling_ratio
        self.query_pad_length = query_pad_length
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.src_num_items = 0
        self.target_num_items = 0
        self.text_embeddings = {}
        self._is_setup = False

    def setup(self, stage: str) -> None:
        if self._is_setup:
            return

        cache_path = Path(self.cache_dir) if self.cache_dir else None
        if cache_path and (cache_path / "uni_cdr_preprocessed.pkl").exists():
            logger.info("Loading cached UniCDR preprocessed data...")
            self._load_cache(cache_path)
        else:
            logger.info("Running UniCDR preprocessing pipeline...")
            self._run_preprocessing()
            if cache_path:
                cache_path.mkdir(parents=True, exist_ok=True)
                self._save_cache(cache_path)

        self._is_setup = True

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_uni_cdr,
        )

    def val_dataloader(self):
        return data_utils.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_uni_cdr,
        )

    def test_dataloader(self):
        return data_utils.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=_collate_uni_cdr,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _run_preprocessing(self):
        # 1. Load interactions
        df = self._load_interactions()
        logger.info(f"Loaded {len(df)} raw interactions")

        # 2. Split by domain category
        df_src = df[df["CATEGORY_SNO"] == self.src_category_sno].copy()
        df_target = df[df["CATEGORY_SNO"] == self.target_category_sno].copy()
        logger.info(f"Source interactions: {len(df_src)}, Target interactions: {len(df_target)}")

        # 3. Build item vocabularies (top N by frequency)
        self.src_item2idx = self._build_item_vocab(df_src, self.src_max_items)
        self.target_item2idx = self._build_item_vocab(df_target, self.target_max_items)
        self.src_num_items = len(self.src_item2idx) - 1  # exclude padding 0
        self.target_num_items = len(self.target_item2idx) - 1
        logger.info(f"Source vocab: {self.src_num_items}, Target vocab: {self.target_num_items}")

        # 4. Encode item indices
        df_src["item_index"] = df_src["ITEM_ID"].astype(str).map(self.src_item2idx).fillna(0).astype(int)
        df_target["item_index"] = df_target["ITEM_ID"].astype(str).map(self.target_item2idx).fillna(0).astype(int)
        df_src = df_src[df_src["item_index"] > 0]
        df_target = df_target[df_target["item_index"] > 0]

        # 5. Classify users
        src_users = set(df_src["USER_ID"].unique())
        target_users = set(df_target["USER_ID"].unique())
        common_users = src_users & target_users
        source_only_users = src_users - target_users
        target_only_users = target_users - src_users
        all_users = src_users | target_users
        logger.info(
            f"Users - common: {len(common_users)}, source_only: {len(source_only_users)}, "
            f"target_only: {len(target_only_users)}"
        )

        # 6. Build per-user sequences
        src_user_seqs = self._build_user_sequences(df_src)
        target_user_seqs = self._build_user_sequences(df_target)

        # 7. Build candidate item pools
        src_cand_items = list(range(1, self.src_num_items + 1))
        target_cand_items = list(range(1, self.target_num_items + 1))

        # 8. Build samples
        rng = random.Random(self.seed)
        train_samples, eval_samples = self._build_samples(
            all_users,
            common_users,
            source_only_users,
            target_only_users,
            src_user_seqs,
            target_user_seqs,
            src_cand_items,
            target_cand_items,
            rng,
        )
        logger.info(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

        # 9. Create datasets
        self.train_dataset = UniCDRTrainDataset(
            samples=train_samples,
            src_num_items=self.src_num_items,
            target_num_items=self.target_num_items,
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
            rng=rng,
        )
        self.val_dataset = UniCDREvalDataset(
            samples=eval_samples,
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
        )
        self.test_dataset = UniCDREvalDataset(
            samples=eval_samples,
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_interactions(self) -> pd.DataFrame:
        parquet_files = sorted(glob(os.path.join(self.interaction_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.interaction_dir}")
        logger.info(f"Loading {len(parquet_files)} parquet files...")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    def _build_item_vocab(self, df: pd.DataFrame, max_items: int) -> dict:
        """Build item vocabulary: item_id -> 1-based index, with 'unknown' -> 0."""
        item_counts = df["ITEM_ID"].value_counts()
        if len(item_counts) > max_items:
            item_counts = item_counts.head(max_items)
        item_ids = item_counts.index.tolist()
        vocab = {str(iid): idx + 1 for idx, iid in enumerate(item_ids)}
        vocab["unknown"] = 0
        return vocab

    def _build_user_sequences(self, df: pd.DataFrame) -> dict:
        """Build per-user sorted interaction sequences."""
        user_seqs = {}
        for user_id, group in df.groupby("USER_ID"):
            group = group.sort_values("TIMESTAMP")
            user_seqs[user_id] = {
                "item_ids": group["item_index"].tolist(),
                "market_ids": group.get("MARKET_ID", pd.Series([1] * len(group))).fillna(1).astype(int).tolist(),
                "category_ids": group.get("CATEGORY_IDX", pd.Series([1] * len(group))).fillna(1).astype(int).tolist(),
            }
        return user_seqs

    def _build_samples(
        self,
        all_users,
        common_users,
        source_only_users,
        target_only_users,
        src_user_seqs,
        target_user_seqs,
        src_cand_items,
        target_cand_items,
        rng,
    ) -> tuple[list[dict], list[dict]]:
        """Build train and eval samples from user sequences."""
        train_samples = []
        eval_samples = []

        empty_seq = {"item_ids": [], "market_ids": [], "category_ids": []}

        for user_id in all_users:
            # Determine user type
            if user_id in common_users:
                user_type = USER_COMMON
            elif user_id in source_only_users:
                user_type = USER_SOURCE
            else:
                user_type = USER_TARGET

            src_seq = src_user_seqs.get(user_id, empty_seq)
            target_seq = target_user_seqs.get(user_id, empty_seq)

            n_src = len(src_seq["item_ids"])
            n_target = len(target_seq["item_ids"])

            # Sample candidates
            n_src_cands = max(int(len(src_cand_items) * self.src_sampling_ratio), 1)
            n_target_cands = max(int(len(target_cand_items) * self.target_sampling_ratio), 1)

            # For eval: hold out last item in target domain as label
            if user_type in (USER_COMMON, USER_TARGET) and n_target >= 2:
                eval_target_seq = {
                    "item_ids": target_seq["item_ids"][:-1],
                    "market_ids": target_seq["market_ids"][:-1],
                    "category_ids": target_seq["category_ids"][:-1],
                }
                target_answer = target_seq["item_ids"][-1]

                # Sample candidates including the answer
                sampled_target_cands = rng.sample(target_cand_items, min(n_target_cands, len(target_cand_items)))
                if target_answer not in sampled_target_cands:
                    sampled_target_cands[0] = target_answer
                target_label_idx = sampled_target_cands.index(target_answer)

                sampled_src_cands = rng.sample(src_cand_items, min(n_src_cands, len(src_cand_items)))
                src_label_idx = 0  # placeholder for source-only eval

                eval_samples.append({
                    "src_item_ids": src_seq["item_ids"],
                    "src_market_ids": src_seq["market_ids"],
                    "src_category_ids": src_seq["category_ids"],
                    "target_item_ids": eval_target_seq["item_ids"],
                    "target_market_ids": eval_target_seq["market_ids"],
                    "target_category_ids": eval_target_seq["category_ids"],
                    "n_src": n_src,
                    "n_target": n_target - 1,
                    "user_type": user_type,
                    "src_cand_items": sampled_src_cands,
                    "target_cand_items": sampled_target_cands,
                    "src_label": src_label_idx,
                    "target_label": target_label_idx,
                })

            # Train sample: use full sequences
            sampled_src_cands = rng.sample(src_cand_items, min(n_src_cands, len(src_cand_items)))
            sampled_target_cands = rng.sample(target_cand_items, min(n_target_cands, len(target_cand_items)))

            # Label = last clicked item placed into candidates
            src_label_idx = 0
            target_label_idx = 0
            if n_src > 0:
                src_answer = src_seq["item_ids"][-1]
                if src_answer not in sampled_src_cands:
                    sampled_src_cands[0] = src_answer
                src_label_idx = sampled_src_cands.index(src_answer)

            if n_target > 0:
                target_answer = target_seq["item_ids"][-1]
                if target_answer not in sampled_target_cands:
                    sampled_target_cands[0] = target_answer
                target_label_idx = sampled_target_cands.index(target_answer)

            train_samples.append({
                "src_item_ids": src_seq["item_ids"],
                "src_market_ids": src_seq["market_ids"],
                "src_category_ids": src_seq["category_ids"],
                "target_item_ids": target_seq["item_ids"],
                "target_market_ids": target_seq["market_ids"],
                "target_category_ids": target_seq["category_ids"],
                "n_src": n_src,
                "n_target": n_target,
                "user_type": user_type,
                "src_cand_items": sampled_src_cands,
                "target_cand_items": sampled_target_cands,
                "src_label": src_label_idx,
                "target_label": target_label_idx,
            })

        return train_samples, eval_samples

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _save_cache(self, cache_path: Path):
        data = {
            "src_num_items": self.src_num_items,
            "target_num_items": self.target_num_items,
            "src_item2idx": self.src_item2idx,
            "target_item2idx": self.target_item2idx,
            "train_samples": self.train_dataset.samples,
            "eval_samples": self.val_dataset.samples,
        }
        with open(cache_path / "uni_cdr_preprocessed.pkl", "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Cached UniCDR data to {cache_path / 'uni_cdr_preprocessed.pkl'}")

    def _load_cache(self, cache_path: Path):
        with open(cache_path / "uni_cdr_preprocessed.pkl", "rb") as f:
            data = pickle.load(f)
        self.src_num_items = data["src_num_items"]
        self.target_num_items = data["target_num_items"]
        self.src_item2idx = data["src_item2idx"]
        self.target_item2idx = data["target_item2idx"]

        rng = random.Random(self.seed)
        self.train_dataset = UniCDRTrainDataset(
            samples=data["train_samples"],
            src_num_items=self.src_num_items,
            target_num_items=self.target_num_items,
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
            rng=rng,
        )
        self.val_dataset = UniCDREvalDataset(
            samples=data["eval_samples"],
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
        )
        self.test_dataset = UniCDREvalDataset(
            samples=data["eval_samples"],
            src_max_len=self.transformer_input_len,
            target_max_len=self.transformer_input_len,
        )
