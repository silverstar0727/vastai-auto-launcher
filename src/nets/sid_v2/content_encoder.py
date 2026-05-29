"""상품 콘텐츠 임베딩 fusion (텍스트 + [이미지 옵션] + 정형 메타).

PoC v1: 텍스트 임베딩(distiluse 512-d) + brand/category embedding lookup.
PoC v2: 이미지(SigLIP2 vision 1152-d) 추가.

산출: 단일 fused embedding (RQ-VAE 입력).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ContentEncoder(nn.Module):
    """
    Args:
        text_dim: distiluse 같은 사전학습 텍스트 임베딩 차원 (0이면 미사용).
        image_dim: SigLIP 같은 비주얼 임베딩 차원 (0이면 미사용).
        num_categories: standard_category 개수 (embedding lookup).
        num_brands:    brand 개수 (embedding lookup, 0 가능).
        cat_emb_dim:   카테고리 embedding 차원.
        brand_emb_dim: brand embedding 차원.
        out_dim:       최종 fused 출력 차원 (RQ-VAE input_dim 과 일치).
        num_price_buckets: 가격 분위 bucket 개수 (0 이면 미사용).
        price_emb_dim: 가격 bucket embedding 차원.
    """

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 0,
        num_categories: int = 1000,
        num_brands: int = 0,
        cat_emb_dim: int = 64,
        brand_emb_dim: int = 32,
        num_price_buckets: int = 0,
        price_emb_dim: int = 16,
        out_dim: int = 512,
        # === v3 신규 (백워드 호환: 기본 0 → 미사용) ===
        num_attribute_values: int = 0,   # value_sno vocab size (e.g., 968+2 for PAD/UNK)
        attribute_emb_dim: int = 32,
        max_attributes_per_item: int = 16,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_categories = num_categories
        self.num_brands = num_brands
        self.num_price_buckets = num_price_buckets
        self.num_attribute_values = num_attribute_values
        self.max_attributes_per_item = max_attributes_per_item

        fused_dim = 0
        if text_dim > 0:
            fused_dim += text_dim
        if image_dim > 0:
            fused_dim += image_dim
        if num_categories > 0:
            self.cat_emb = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=0)
            fused_dim += cat_emb_dim
        if num_brands > 0:
            self.brand_emb = nn.Embedding(num_brands + 1, brand_emb_dim, padding_idx=0)
            fused_dim += brand_emb_dim
        if num_price_buckets > 0:
            self.price_emb = nn.Embedding(num_price_buckets + 1, price_emb_dim, padding_idx=0)
            fused_dim += price_emb_dim
        if num_attribute_values > 0:
            # multi-attribute: 각 item 은 N(<=max) 개 attribute value 가짐 → mean pooling
            self.attribute_emb = nn.Embedding(
                num_attribute_values + 1, attribute_emb_dim, padding_idx=0
            )
            fused_dim += attribute_emb_dim

        if fused_dim == 0:
            raise ValueError("ContentEncoder: at least one modality required")

        # projection to out_dim (RQ-VAE 입력에 맞춤)
        self.proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, out_dim),
        )
        self.fused_dim = fused_dim
        self.out_dim = out_dim

    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
        category_id: Optional[torch.Tensor] = None,
        brand_id: Optional[torch.Tensor] = None,
        price_bucket: Optional[torch.Tensor] = None,
        attribute_ids: Optional[torch.Tensor] = None,  # (B, max_attributes) — PAD=0
    ) -> torch.Tensor:
        parts = []
        if self.text_dim > 0 and text_emb is not None:
            parts.append(text_emb)
        if self.image_dim > 0 and image_emb is not None:
            parts.append(image_emb)
        if self.num_categories > 0 and category_id is not None:
            parts.append(self.cat_emb(category_id))
        if self.num_brands > 0 and brand_id is not None:
            parts.append(self.brand_emb(brand_id))
        if self.num_price_buckets > 0 and price_bucket is not None:
            parts.append(self.price_emb(price_bucket))
        if self.num_attribute_values > 0 and attribute_ids is not None:
            # attribute_ids: (B, K) — PAD=0 인 위치는 padding_idx 로 0 벡터 자동 처리
            attr_embs = self.attribute_emb(attribute_ids)  # (B, K, attr_dim)
            mask = (attribute_ids != 0).float().unsqueeze(-1)  # (B, K, 1)
            denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — 최소 1개
            pooled = (attr_embs * mask).sum(dim=1) / denom  # (B, attr_dim) mean pooling
            parts.append(pooled)
        if not parts:
            raise ValueError("ContentEncoder.forward: no inputs provided")
        fused = torch.cat(parts, dim=-1)
        return self.proj(fused)
