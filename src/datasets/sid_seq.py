"""TIGER-lite sequential DataModule.

전제:
  - meta/active_items.parquet
  - meta/item_to_sid.parquet  (SID Tokenizer 학습 산출물)
    컬럼: goods_sno, sid_0, sid_1, sid_2 (각 레벨 codebook index, 0-based)
  - interactions/{train,val,test}/*.parquet

토큰화:
  - 시퀀스의 각 item 을 SID 토큰 L 개로 flatten → encoder 입력 (T_enc = N × L)
  - target: 마지막 item 의 SID (decoder)
  - behavior_id 는 item 별로 broadcast (L 번 반복)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader

from nets.tiger_lite import build_sid_vocab

BEHAVIOR_TO_ID = {"click": 1, "like": 2, "cart": 3, "purchase": 4}
PAD_ID, BOS_ID, SEP_ID, EOS_ID = 0, 1, 2, 3


def _absolute_sid(level_codes: np.ndarray, offsets: List[int]) -> np.ndarray:
    """(N, L) per-level codes → (N, L) absolute token ids."""
    off = np.array(offsets, dtype=np.int64)[None, :]
    return level_codes + off


class _SIDStore:
    """goods_sno → absolute SID tuple (length L)."""

    def __init__(self, item_to_sid_df: pd.DataFrame, codebook_sizes: Tuple[int, ...]):
        offsets, total_vocab = build_sid_vocab(codebook_sizes)
        self.offsets = offsets
        self.total_vocab = total_vocab
        self.L = len(codebook_sizes)
        # 검증: per-level codes 가 [0, codebook_size) 안에 있는지
        codes = item_to_sid_df[[f"sid_{l}" for l in range(self.L)]].values.astype(np.int64)
        for l in range(self.L):
            assert codes[:, l].max() < codebook_sizes[l], f"sid_{l} 가 codebook 범위 밖"
        abs_codes = _absolute_sid(codes, offsets)
        self.sno_to_sid: Dict[int, np.ndarray] = {
            int(sno): abs_codes[i] for i, sno in enumerate(item_to_sid_df["goods_sno"].astype(int))
        }

    def lookup(self, snos: List[int]) -> Optional[np.ndarray]:
        out = []
        for s in snos:
            sid = self.sno_to_sid.get(int(s))
            if sid is None:
                return None
            out.append(sid)
        return np.stack(out) if out else None

    def all_item_sids(self) -> torch.Tensor:
        arr = np.stack(list(self.sno_to_sid.values()))
        return torch.from_numpy(arr)


class _SIDSeqTrainDataset(Dataset):
    """학습: encoder = user history items (SID flat), decoder = next item SID."""

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "enc_tokens": torch.from_numpy(s["enc_tokens"]),
            "enc_beh": torch.from_numpy(s["enc_beh"]),
            "enc_pos": torch.from_numpy(s["enc_pos"]),
            "enc_order_pos": torch.from_numpy(s["enc_order_pos"]),
            "enc_mask": torch.from_numpy(s["enc_mask"]).bool(),
            "dec_input": torch.from_numpy(s["dec_input"]),
            "dec_target": torch.from_numpy(s["dec_target"]),
            "dec_pos": torch.from_numpy(s["dec_pos"]),
            "target_behavior": torch.tensor(s["target_behavior"], dtype=torch.long),
        }


class _SIDSeqParquetDataset(Dataset):
    """Disk-backed: memory-mapped Arrow Table 의 row 를 lazy 하게 읽고 pad.

    파일 schema 는 `TIGERLiteDataModule._build_samples_to_parquet` 참고.
    Padding 은 __getitem__ 에서 수행하여 disk 사용량 축소 (~60% 감소 대비 pre-pad).
    """

    def __init__(
        self,
        table: pa.Table,
        max_history_items: int,
        L: int,
        use_order_pos: bool,
    ):
        self.table = table
        self.max_history_items = max_history_items
        self.L = L
        self.max_T_enc = max_history_items * L
        self.use_order_pos = use_order_pos
        # pre-resolve columns to ChunkedArray for fast row access
        self._c_hist_sids = table.column("hist_sids")
        self._c_hist_evs = table.column("hist_evs")
        self._c_hist_order = table.column("hist_order_pos")
        self._c_dec_input = table.column("dec_input")
        self._c_dec_target = table.column("dec_target")
        self._c_dec_pos = table.column("dec_pos")
        self._c_target_beh = table.column("target_behavior")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        hist_sids = np.asarray(self._c_hist_sids[idx].as_py(), dtype=np.int64)
        hist_evs = np.asarray(self._c_hist_evs[idx].as_py(), dtype=np.int64)
        n_hist = len(hist_evs)

        enc_tokens = np.zeros(self.max_T_enc, dtype=np.int64)
        enc_beh = np.zeros(self.max_T_enc, dtype=np.int64)
        enc_order_pos = np.zeros(self.max_T_enc, dtype=np.int64)

        # right-align: 가장 최근 history 가 max_T_enc 끝으로
        n_clamp = min(n_hist, self.max_history_items)
        n_tokens = n_clamp * self.L
        if n_tokens > 0:
            enc_tokens[-n_tokens:] = hist_sids[-n_tokens:]
            enc_beh[-n_tokens:] = np.repeat(hist_evs[-n_clamp:], self.L)
            if self.use_order_pos:
                hist_op = np.asarray(self._c_hist_order[idx].as_py(), dtype=np.int64)
                if len(hist_op) > 0:
                    enc_order_pos[-n_tokens:] = np.repeat(hist_op[-n_clamp:], self.L)
        enc_pos = np.arange(self.max_T_enc, dtype=np.int64)
        enc_mask = (enc_tokens > 0)

        dec_input = np.asarray(self._c_dec_input[idx].as_py(), dtype=np.int64)
        dec_target = np.asarray(self._c_dec_target[idx].as_py(), dtype=np.int64)
        dec_pos = np.asarray(self._c_dec_pos[idx].as_py(), dtype=np.int64)
        target_behavior = self._c_target_beh[idx].as_py()

        return {
            "enc_tokens": torch.from_numpy(enc_tokens),
            "enc_beh": torch.from_numpy(enc_beh),
            "enc_pos": torch.from_numpy(enc_pos),
            "enc_order_pos": torch.from_numpy(enc_order_pos),
            "enc_mask": torch.from_numpy(enc_mask),
            "dec_input": torch.from_numpy(dec_input),
            "dec_target": torch.from_numpy(dec_target),
            "dec_pos": torch.from_numpy(dec_pos),
            "target_behavior": torch.tensor(target_behavior, dtype=torch.long),
        }


class TIGERLiteDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/home/jeongmindo/projects/ably/data/reco_experiments/v2_full",
        codebook_sizes: tuple = (2048, 1024, 512),
        max_history_items: int = 50,
        min_history_items: int = 5,
        max_user_history: int = 500,        # augment 원본 시퀀스 길이
        augment_factor: int = 8,
        batch_size: int = 128,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        train_days: int = 30,
        val_days: int = 7,
        test_days: int = 16,
        seed: int = 42,
        # === v7 신규 (백워드 호환: False → 기존 동작) ===
        use_order_pos: bool = False,             # 같은 ts → 같은 order position (cart/outfit 신호)
        use_popularity_bias_augment: bool = False,  # augment end position 을 inv-freq 가중 샘플링
        item_to_sid_path: Optional[str] = None,  # None 이면 meta/item_to_sid.parquet 기본
        # === disk cache (None 이면 in-memory 동작) ===
        samples_cache_dir: Optional[str] = None,
        samples_chunk_size: int = 100_000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base = Path(base_dir)
        self.store: Optional[_SIDStore] = None
        self._train_samples = None
        self._val_samples = None
        self._test_samples = None
        self._train_table: Optional[pa.Table] = None
        self._val_table: Optional[pa.Table] = None
        self._item_freq: Dict[int, int] = {}

    @property
    def num_behaviors(self):
        return max(BEHAVIOR_TO_ID.values()) + 1

    @property
    def total_vocab(self):
        return self.store.total_vocab if self.store else 0

    def setup(self, stage: Optional[str] = None):
        if self.store is not None:
            return
        if self.hparams.item_to_sid_path:
            _p = Path(self.hparams.item_to_sid_path)
            sid_path = _p if _p.is_absolute() else self.base / _p
        else:
            sid_path = self.base / "meta" / "item_to_sid.parquet"
        item_sid = pd.read_parquet(sid_path)
        item_sid["goods_sno"] = pd.to_numeric(item_sid["goods_sno"], errors="coerce").astype("int64")
        self.store = _SIDStore(item_sid, tuple(self.hparams.codebook_sizes))

        train_files = sorted((self.base / "interactions" / "train").glob("*.parquet"))[-self.hparams.train_days:]
        val_files = sorted((self.base / "interactions" / "val").glob("*.parquet"))[-self.hparams.val_days:]
        test_files = sorted((self.base / "interactions" / "test").glob("*.parquet"))[-self.hparams.test_days:]

        # v7: popularity bias 용 item 빈도 pre-pass
        if self.hparams.use_popularity_bias_augment:
            print("[TIGER-DM] popularity bias — item freq pre-pass...", flush=True)
            from collections import Counter
            counter: Counter = Counter()
            for fp in train_files:
                df = pd.read_parquet(fp, columns=["goods_sno"])
                counter.update(df["goods_sno"].astype("int64").to_numpy().tolist())
            self._item_freq = dict(counter)
            print(f"[TIGER-DM]   freq: {len(self._item_freq):,} unique items", flush=True)

        if self.hparams.samples_cache_dir:
            cache_root = Path(self.hparams.samples_cache_dir)
            key = self._hash_cache_key()
            cache_dir = cache_root / key
            train_pq = cache_dir / "train.parquet"
            val_pq = cache_dir / "val.parquet"
            print(f"[TIGER-DM] disk cache: {cache_dir}", flush=True)
            if not (train_pq.exists() and val_pq.exists()):
                cache_dir.mkdir(parents=True, exist_ok=True)
                # meta
                meta = {
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool, type(None)))
                        else list(v) if isinstance(v, (tuple, list))
                        else str(v)
                    )
                    for k, v in dict(self.hparams).items()
                }
                (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                self._build_samples_to_parquet(train_files, augment=True, out_path=train_pq)
                self._build_samples_to_parquet(val_files, augment=False, out_path=val_pq)
            else:
                print(f"[TIGER-DM]   cache hit — skip build", flush=True)
            self._train_table = pq.read_table(train_pq, memory_map=True)
            self._val_table = pq.read_table(val_pq, memory_map=True)
            print(
                f"[TIGER-DM] loaded — train={len(self._train_table):,}, "
                f"val={len(self._val_table):,}",
                flush=True,
            )
        else:
            self._train_samples = self._build_samples(train_files, augment=True)
            self._val_samples = self._build_samples(val_files, augment=False)

        # test: stage 인자가 "test" 일 때만 빌드 (leave-last-out, augment=False → user당 마지막 1 sample)
        if stage == "test" and self._test_samples is None and test_files:
            print(f"[TIGER-DM] building test samples (leave-last-out) — {len(test_files)} files", flush=True)
            self._test_samples = self._build_samples(test_files, augment=False)
            # multi-interest-prod 의 max_test_samples=200,000 과 통일 (random sample)
            MAX_TEST = 200_000
            n_built = len(self._test_samples)
            if n_built > MAX_TEST:
                rng = np.random.default_rng(self.hparams.seed)
                idx = sorted(rng.choice(n_built, size=MAX_TEST, replace=False).tolist())
                self._test_samples = [self._test_samples[i] for i in idx]
                print(f"[TIGER-DM] test capped: {n_built:,} → {MAX_TEST:,} (random sample, seed={self.hparams.seed})", flush=True)
            else:
                print(f"[TIGER-DM] test samples = {n_built:,} (under {MAX_TEST:,} cap)", flush=True)

    def _build_samples(self, files: List[Path], augment: bool = False) -> List[Dict[str, np.ndarray]]:
        rows = []
        for fp in files:
            df = pd.read_parquet(fp, columns=["user_code", "goods_sno", "event", "ts"])
            rows.append(df)
        if not rows:
            return []
        all_df = pd.concat(rows, ignore_index=True).dropna()
        all_df["goods_sno"] = pd.to_numeric(all_df["goods_sno"], errors="coerce").astype("int64")
        all_df["ts"] = pd.to_numeric(all_df["ts"], errors="coerce").astype("int64")
        all_df = all_df.dropna(subset=["goods_sno", "ts"]).sort_values(
            ["user_code", "ts"], kind="stable"
        )

        L = self.store.L
        max_T_enc = self.hparams.max_history_items * L
        samples: List[Dict[str, np.ndarray]] = []
        rng = np.random.default_rng(self.hparams.seed)

        n_users = all_df["user_code"].nunique()
        print(
            f"[build_samples] augment={augment} rows={len(all_df):,} users={n_users:,} "
            f"order_pos={self.hparams.use_order_pos} pop_bias={self.hparams.use_popularity_bias_augment}",
            flush=True,
        )
        _ui = 0
        for uc, g in all_df.groupby("user_code", sort=False):
            _ui += 1
            if _ui % 300_000 == 0:
                print(f"[build_samples]   {_ui:,}/{n_users:,} users, samples={len(samples):,}", flush=True)
            snos = g["goods_sno"].tolist()[-self.hparams.max_user_history:]
            evs = g["event"].tolist()[-self.hparams.max_user_history:]
            tss = g["ts"].tolist()[-self.hparams.max_user_history:]
            if len(snos) < self.hparams.min_history_items + 1:
                continue

            valid_sids = self.store.lookup(snos)
            if valid_sids is None:
                continue

            # v7: order position — 같은 ts 는 같은 order position (cart/outfit 동시구매 신호)
            if self.hparams.use_order_pos:
                order_pos_per_item = np.zeros(len(snos), dtype=np.int64)
                last_ts = -1
                cur_pos = -1
                for i, t in enumerate(tss):
                    if t != last_ts:
                        cur_pos += 1
                        last_ts = t
                    order_pos_per_item[i] = cur_pos
            else:
                order_pos_per_item = None

            n_pos = len(snos)
            if augment:
                k = min(self.hparams.augment_factor, n_pos - 1)
                # v7: popularity bias — end position 을 inv-freq 가중
                if self.hparams.use_popularity_bias_augment and self._item_freq:
                    candidate_idx = np.arange(1, n_pos)
                    weights = np.array(
                        [
                            1.0 / np.sqrt(max(self._item_freq.get(int(snos[i]), 1), 1))
                            for i in candidate_idx
                        ],
                        dtype=np.float64,
                    )
                    weights = weights / weights.sum()
                    end_positions = rng.choice(
                        candidate_idx, size=k, replace=False, p=weights
                    ).tolist()
                else:
                    end_positions = rng.choice(
                        np.arange(1, n_pos), size=k, replace=False
                    ).tolist()
            else:
                end_positions = [n_pos - 1]

            for end in end_positions:
                target_sid = valid_sids[end]
                history_sids = valid_sids[max(0, end - self.hparams.max_history_items):end]
                history_evs = evs[max(0, end - self.hparams.max_history_items):end]
                if len(history_sids) < self.hparams.min_history_items:
                    continue
                target_event = evs[end]
                n_hist = len(history_sids)

                enc_tokens = np.zeros(max_T_enc, dtype=np.int64)
                enc_beh = np.zeros(max_T_enc, dtype=np.int64)
                enc_order_pos = np.zeros(max_T_enc, dtype=np.int64)
                flat = history_sids.flatten()
                flat_beh = np.repeat(
                    [BEHAVIOR_TO_ID.get(e, 0) for e in history_evs], L
                ).astype(np.int64)
                enc_tokens[-n_hist * L:] = flat
                enc_beh[-n_hist * L:] = flat_beh
                if order_pos_per_item is not None:
                    hist_order = order_pos_per_item[max(0, end - self.hparams.max_history_items):end]
                    # 같은 SID 의 L 개 토큰 → 같은 order position 공유
                    enc_order_pos[-n_hist * L:] = np.repeat(hist_order, L)
                enc_pos = np.arange(max_T_enc, dtype=np.int64)
                enc_mask = (enc_tokens > 0).astype(np.int64)
                dec_input = np.concatenate([[BOS_ID], target_sid]).astype(np.int64)
                dec_target = np.concatenate([target_sid, [EOS_ID]]).astype(np.int64)
                dec_pos = np.arange(dec_input.shape[0], dtype=np.int64)

                samples.append({
                    "enc_tokens": enc_tokens,
                    "enc_beh": enc_beh,
                    "enc_pos": enc_pos,
                    "enc_order_pos": enc_order_pos,
                    "enc_mask": enc_mask,
                    "dec_input": dec_input,
                    "dec_target": dec_target,
                    "dec_pos": dec_pos,
                    "target_behavior": BEHAVIOR_TO_ID.get(target_event, 0),
                })
        print(f"[build_samples] augment={augment} 완료 — samples={len(samples):,}", flush=True)
        return samples

    def _hash_cache_key(self) -> str:
        keys = {
            "item_to_sid_path": str(self.hparams.item_to_sid_path),
            "codebook_sizes": list(self.hparams.codebook_sizes),
            "train_days": self.hparams.train_days,
            "val_days": self.hparams.val_days,
            "max_history_items": self.hparams.max_history_items,
            "min_history_items": self.hparams.min_history_items,
            "max_user_history": self.hparams.max_user_history,
            "augment_factor": self.hparams.augment_factor,
            "use_order_pos": self.hparams.use_order_pos,
            "use_popularity_bias_augment": self.hparams.use_popularity_bias_augment,
            "seed": self.hparams.seed,
        }
        return hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]

    def _build_samples_to_parquet(
        self, files: List[Path], augment: bool, out_path: Path
    ) -> int:
        """Polars lazy sort → pyarrow streaming groupby → parquet stream-write.

        Memory 는 폴라스 streaming engine + pyarrow batch 단위로 const.
        스키마 (variable-length list 컬럼, padding 은 __getitem__ 에서):
          hist_sids: list<int64>  - history items × L (flat)
          hist_evs:  list<int64>  - history behaviors (length n_hist)
          hist_order_pos: list<int64> - order pos (use_order_pos=True 일 때만)
          dec_input: list<int64>  - (1 + L,)
          dec_target: list<int64> - (L + 1,)
          dec_pos: list<int64>    - (1 + L,)
          target_behavior: int64
        """
        import duckdb

        if not files:
            return 0

        # 1) DuckDB out-of-core sort (spill to disk) → sorted parquet
        sorted_path = out_path.parent / f"_sorted_{out_path.stem}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if sorted_path.exists():
            sorted_path.unlink()
        print(
            f"[build_parquet] augment={augment} duckdb sort {len(files)} files "
            f"→ {sorted_path.name}",
            flush=True,
        )
        con = duckdb.connect(":memory:")
        # 메모리 cap + spill dir — 시스템 OOM 방지 (123GB RAM 환경)
        con.execute("SET memory_limit='30GB'")
        spill_dir = out_path.parent / ".duckdb_spill"
        spill_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{spill_dir}'")
        # 보호: progress 가 늦으면 큰 작업이므로 진행 표시
        con.execute("PRAGMA enable_progress_bar")

        files_sql = ", ".join(f"'{str(f)}'" for f in files)
        con.execute(
            f"""
            COPY (
                SELECT
                    user_code,
                    CAST(goods_sno AS BIGINT) AS goods_sno,
                    event,
                    CAST(ts AS BIGINT) AS ts
                FROM read_parquet([{files_sql}])
                WHERE user_code IS NOT NULL
                  AND goods_sno IS NOT NULL
                  AND ts IS NOT NULL
                  AND event IS NOT NULL
                ORDER BY user_code, ts
            )
            TO '{sorted_path}' (FORMAT PARQUET, ROW_GROUP_SIZE 200000, COMPRESSION ZSTD)
            """
        )
        con.close()
        # spill dir 청소
        import shutil as _sh
        try:
            _sh.rmtree(spill_dir)
        except OSError:
            pass
        print(
            f"[build_parquet]   sorted: {sorted_path.stat().st_size / 1e9:.2f}GB",
            flush=True,
        )

        # 2) pyarrow batched iteration → streaming groupby by user_code
        schema = pa.schema([
            ("hist_sids", pa.list_(pa.int64())),
            ("hist_evs", pa.list_(pa.int64())),
            ("hist_order_pos", pa.list_(pa.int64())),
            ("dec_input", pa.list_(pa.int64())),
            ("dec_target", pa.list_(pa.int64())),
            ("dec_pos", pa.list_(pa.int64())),
            ("target_behavior", pa.int64()),
        ])
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")

        chunk = self.hparams.samples_chunk_size
        buf: Dict[str, List] = {n: [] for n in schema.names}
        n_total = 0
        n_users_done = 0
        rng = np.random.default_rng(self.hparams.seed)
        L = self.store.L

        def flush():
            nonlocal buf
            if not buf["hist_sids"]:
                return
            tbl = pa.Table.from_pydict(buf, schema=schema)
            writer.write_table(tbl)
            buf = {n: [] for n in schema.names}

        def process_user(snos_list, evs_list, tss_list):
            """단일 user history → 0..k 개 sample buf 에 append."""
            nonlocal n_total
            snos = snos_list[-self.hparams.max_user_history:]
            evs = evs_list[-self.hparams.max_user_history:]
            tss = tss_list[-self.hparams.max_user_history:]
            if len(snos) < self.hparams.min_history_items + 1:
                return
            valid_sids = self.store.lookup(snos)
            if valid_sids is None:
                return

            if self.hparams.use_order_pos:
                order_pos_per_item = np.zeros(len(snos), dtype=np.int64)
                last_ts = -1
                cur_pos = -1
                for i, t in enumerate(tss):
                    if t != last_ts:
                        cur_pos += 1
                        last_ts = t
                    order_pos_per_item[i] = cur_pos
            else:
                order_pos_per_item = None

            n_pos = len(snos)
            if augment:
                k = min(self.hparams.augment_factor, n_pos - 1)
                if self.hparams.use_popularity_bias_augment and self._item_freq:
                    candidate_idx = np.arange(1, n_pos)
                    weights = np.array(
                        [
                            1.0 / np.sqrt(max(self._item_freq.get(int(snos[i]), 1), 1))
                            for i in candidate_idx
                        ],
                        dtype=np.float64,
                    )
                    weights = weights / weights.sum()
                    end_positions = rng.choice(
                        candidate_idx, size=k, replace=False, p=weights
                    ).tolist()
                else:
                    end_positions = rng.choice(
                        np.arange(1, n_pos), size=k, replace=False
                    ).tolist()
            else:
                end_positions = [n_pos - 1]

            for end in end_positions:
                target_sid = valid_sids[end]
                hist_start = max(0, end - self.hparams.max_history_items)
                history_sids = valid_sids[hist_start:end]
                history_evs = evs[hist_start:end]
                if len(history_sids) < self.hparams.min_history_items:
                    continue
                target_event = evs[end]

                buf["hist_sids"].append(history_sids.flatten().tolist())
                buf["hist_evs"].append(
                    [BEHAVIOR_TO_ID.get(e, 0) for e in history_evs]
                )
                if order_pos_per_item is not None:
                    buf["hist_order_pos"].append(
                        order_pos_per_item[hist_start:end].tolist()
                    )
                else:
                    buf["hist_order_pos"].append([])
                buf["dec_input"].append([BOS_ID] + target_sid.tolist())
                buf["dec_target"].append(target_sid.tolist() + [EOS_ID])
                buf["dec_pos"].append(list(range(1 + len(target_sid))))
                buf["target_behavior"].append(BEHAVIOR_TO_ID.get(target_event, 0))
                n_total += 1
                if n_total % chunk == 0:
                    flush()

        pf = pq.ParquetFile(sorted_path)
        print(f"[build_parquet]   pyarrow batch stream, rows={pf.metadata.num_rows:,}", flush=True)
        cur_user = None
        cur_snos: List[int] = []
        cur_evs: List[str] = []
        cur_tss: List[int] = []
        for batch in pf.iter_batches(batch_size=500_000,
                                     columns=["user_code", "goods_sno", "event", "ts"]):
            uc_arr = batch.column("user_code").to_pylist()
            sno_arr = batch.column("goods_sno").to_pylist()
            ev_arr = batch.column("event").to_pylist()
            ts_arr = batch.column("ts").to_pylist()
            for i in range(len(uc_arr)):
                uc = uc_arr[i]
                if uc != cur_user:
                    if cur_user is not None:
                        process_user(cur_snos, cur_evs, cur_tss)
                        n_users_done += 1
                        if n_users_done % 300_000 == 0:
                            print(
                                f"[build_parquet]   {n_users_done:,} users done, "
                                f"samples={n_total:,}",
                                flush=True,
                            )
                    cur_user = uc
                    cur_snos, cur_evs, cur_tss = [], [], []
                cur_snos.append(sno_arr[i])
                cur_evs.append(ev_arr[i])
                cur_tss.append(ts_arr[i])
        # 마지막 user flush
        if cur_user is not None:
            process_user(cur_snos, cur_evs, cur_tss)
            n_users_done += 1

        flush()
        writer.close()
        # 임시 sorted 파일 제거
        try:
            sorted_path.unlink()
        except OSError:
            pass
        sz = out_path.stat().st_size / 1e9
        print(
            f"[build_parquet] {out_path.name} 완료 — users={n_users_done:,}, "
            f"samples={n_total:,}, size={sz:.2f}GB",
            flush=True,
        )
        return n_total

    def train_dataloader(self):
        if self._train_table is not None:
            ds = _SIDSeqParquetDataset(
                self._train_table,
                max_history_items=self.hparams.max_history_items,
                L=self.store.L,
                use_order_pos=self.hparams.use_order_pos,
            )
        else:
            ds = _SIDSeqTrainDataset(self._train_samples)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        if self._val_table is not None:
            ds = _SIDSeqParquetDataset(
                self._val_table,
                max_history_items=self.hparams.max_history_items,
                L=self.store.L,
                use_order_pos=self.hparams.use_order_pos,
            )
        else:
            ds = _SIDSeqTrainDataset(self._val_samples)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        # leave-last-out test (user당 1 sample, multi-interest-prod 평가 방식과 통일)
        if self._test_samples is None:
            raise RuntimeError("test_samples 가 빌드되지 않았습니다. trainer.test(...) 호출 또는 setup(stage='test') 필요")
        ds = _SIDSeqTrainDataset(self._test_samples)
        return DataLoader(ds, batch_size=self.hparams.eval_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)
