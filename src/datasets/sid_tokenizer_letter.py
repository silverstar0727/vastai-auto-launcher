"""SID Tokenizer DataModule with frozen CF teacher embeddings (LETTER).

기존 `SIDTokenizerDataModule` 을 그대로 사용하면서, item feature store 에 CF
teacher embedding lookup 을 덧붙인다. CF embedding 은 미리 추출된 파일을 읽어
frozen tensor 로 보관.

Prerequisite:
  meta/cf_embeddings.pt — torch.save 로 저장된 dict:
    {
      "goods_sno": np.ndarray[int64]   shape (N,)
      "emb":       np.ndarray|tensor   shape (N, cf_dim)
    }
  scripts/extract_cf_embeddings.py 로 HSTU/SASRec ckpt 에서 추출.

기존 `SIDTokenizerDataModule` 에 영향 없음 — 본 모듈을 사용하지 않으면 동일 동작.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .sid_tokenizer import SIDTokenizerDataModule, _ItemFeatureStore


class _ItemFeatureStoreWithCF:
    """기존 store wrapper — get() 결과에 'cf_emb' key 추가."""

    def __init__(
        self,
        base: _ItemFeatureStore,
        cf_emb: torch.Tensor,
        sno_to_cf_row: Dict[int, int],
    ):
        self._base = base
        self.cf_emb = cf_emb  # (N, cf_dim) float32
        self.sno_to_cf_row = sno_to_cf_row
        self.cf_dim = cf_emb.size(1)

    # 부모 store 의 인터페이스 그대로 통과
    @property
    def goods_index(self):
        return self._base.goods_index

    @property
    def text_emb(self):
        return self._base.text_emb

    @property
    def text_emb_index(self):
        return self._base.text_emb_index

    def get(self, goods_sno: int) -> Dict[str, torch.Tensor]:
        d = self._base.get(goods_sno)
        row = self.sno_to_cf_row.get(int(goods_sno))
        if row is not None:
            d["cf_emb"] = self.cf_emb[row].clone()
        else:
            # cold item: CF embedding 없음 → 영벡터 (학습 시 노이즈 부담)
            d["cf_emb"] = torch.zeros(self.cf_dim, dtype=torch.float32)
        return d


class SIDTokenizerLetterDataModule(SIDTokenizerDataModule):
    """LETTER 변형 DataModule — CF teacher embedding 자동 로드 + batch 에 'cf_emb' 포함."""

    def __init__(
        self,
        cf_emb_path: str = "meta/cf_embeddings.pt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters("cf_emb_path")

    def setup(self, stage: Optional[str] = None):
        if self.store is not None:
            return
        super().setup(stage)

        # CF embedding 로드
        cf_path = Path(self.hparams.cf_emb_path)
        if not cf_path.is_absolute():
            cf_path = self.base / cf_path
        if not cf_path.exists():
            raise FileNotFoundError(
                f"[LETTER] CF teacher embedding 파일이 없습니다: {cf_path}\n"
                f"scripts/extract_cf_embeddings.py 로 먼저 추출하세요."
            )
        bundle = torch.load(cf_path, map_location="cpu", weights_only=False)
        snos = np.asarray(bundle["goods_sno"]).astype(np.int64)
        emb = bundle["emb"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.from_numpy(np.asarray(emb))
        emb = emb.float()
        sno_to_row = {int(s): i for i, s in enumerate(snos)}

        print(
            f"[LETTER-DM] cf_emb loaded — {emb.size(0):,} items × {emb.size(1)} dim "
            f"({cf_path})"
        )

        # store wrap
        self.store = _ItemFeatureStoreWithCF(
            base=self.store, cf_emb=emb, sno_to_cf_row=sno_to_row
        )
        self.cf_emb_dim = emb.size(1)
