"""HSTU two-tower architecture.

User tower:  multi-behavior 토큰 시퀀스 → HSTU encoder → multi-interest head → K user vectors
Item tower:  item embedding (학습 가능 + content embedding mix) → item vector

Score:  user_K · item_D  (max over K  또는  sum-of-rank-weighted, downstream)

토큰 schema (sequence input):
  - item_id              : (B, T) long
  - behavior_type        : (B, T) long  ∈ {click=1, like=2, cart=3, purchase=4, pad=0}
  - position             : (B, T) long  — positional encoding 또는 time bucket
  - mask                 : (B, T) bool

content feature (옵션, item tower 보강):
  - text_emb_lookup table (사전학습 distiluse 등) → item tower 에 add
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .hstu_block import HSTUEncoder
from .multi_interest_head import MultiInterestHead


class HSTUTwoTowerNet(nn.Module):
    """
    Args:
        num_items:             vocab (active items 수 + 1 for pad)
        num_behaviors:         이벤트 수 (4 + 1 pad = 5)
        max_seq_len:           최대 시퀀스 길이
        dim:                   hidden dim
        num_layers:            HSTU layer 수
        num_heads:              attention head
        num_interests:         multi-interest K
        item_content_dim:      옵션 content embedding 차원 (0 이면 미사용)
        dropout:               dropout
    """

    def __init__(
        self,
        num_items: int,
        num_behaviors: int = 5,
        max_seq_len: int = 200,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        num_interests: int = 4,
        attention_dim: int = 64,
        item_content_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_interests = num_interests

        # embeddings (token + behavior + position) → sum
        self.item_emb = nn.Embedding(num_items, dim, padding_idx=0)
        self.behavior_emb = nn.Embedding(num_behaviors, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # content embedding (옵션 — 사전학습 distiluse 등을 LayerNorm + Linear 로 흡수)
        if item_content_dim > 0:
            self.content_proj = nn.Sequential(
                nn.LayerNorm(item_content_dim),
                nn.Linear(item_content_dim, dim, bias=False),
            )
        else:
            self.content_proj = None

        self.input_drop = nn.Dropout(dropout)
        self.encoder = HSTUEncoder(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout, causal=True,
        )
        self.user_head = MultiInterestHead(
            dim=dim, num_interests=num_interests, attention_dim=attention_dim,
        )

    def _embed_items(
        self,
        item_ids: torch.Tensor,
        content_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B, T) item ids → (B, T, D) item embeddings (id + optional content)."""
        e = self.item_emb(item_ids)
        if self.content_proj is not None and content_emb is not None:
            e = e + self.content_proj(content_emb)
        return e

    def encode_user(
        self,
        item_ids: torch.Tensor,            # (B, T)
        behavior_ids: torch.Tensor,        # (B, T)
        positions: torch.Tensor,           # (B, T)
        mask: torch.Tensor,                # (B, T) bool
        content_emb: Optional[torch.Tensor] = None,  # (B, T, content_dim) optional
    ) -> torch.Tensor:
        """user → (B, K, D) interest 벡터."""
        h = (
            self._embed_items(item_ids, content_emb)
            + self.behavior_emb(behavior_ids)
            + self.pos_emb(positions.clamp(max=self.max_seq_len - 1))
        )
        h = self.input_drop(h)
        h = self.encoder(h, attn_mask=mask)
        return self.user_head(h, mask=mask)

    def encode_item(
        self,
        item_ids: torch.Tensor,                       # (N,) or (B, N)
        content_emb: Optional[torch.Tensor] = None,   # 같은 shape + last dim content
    ) -> torch.Tensor:
        """item → (N, D) 또는 (B, N, D) item 벡터."""
        return self._embed_items(item_ids, content_emb)

    def score(
        self,
        user_interests: torch.Tensor,    # (B, K, D)
        item_vec: torch.Tensor,          # (B, N, D) candidate items
    ) -> torch.Tensor:
        """user interest 와 candidate items 의 max-over-K dot product."""
        # (B, K, D) · (B, N, D)→ (B, K, N), max over K → (B, N)
        scores = torch.einsum("bkd,bnd->bkn", user_interests, item_vec)
        return scores.max(dim=1).values
