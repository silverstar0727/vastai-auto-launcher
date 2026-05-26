"""HSTU (Hierarchical Sequential Transduction Units) — Meta, ICML 2024.

ref: "Actions Speak Louder than Words" (Zhai et al., 2024) — arXiv 2402.17152

기존 self-attention 의 변형:
  - Pointwise projection: U, V, Q, K 4-way 분리 + SiLU
  - Spatial aggregation: rel-attention via Q·K^T (causal/normalized)
  - Gating: u_pre = U(x);  out = norm( gate · pointwise_aggregate )
  - Layer-norm 위치를 ablation 친화적으로

본 구현은 단순화된 버전 — torch.nn.functional.scaled_dot_product_attention 활용해
FlashAttention 백엔드를 자동으로 사용 (PyTorch 2.x).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSTUBlock(nn.Module):
    """단일 HSTU layer.

    Args:
        dim:        모델 hidden dim
        num_heads:  attention head
        ff_dim:     gate 계산용 ff
        dropout:    attention / ff dropout
        causal:     causal mask 적용 (sequential recsys 기본 True)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        ff = ff_dim or 4 * dim

        # 4-way pointwise projection (U, V, Q, K)
        self.uvqk_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        # gate path: linear → silu → linear
        self.gate = nn.Sequential(
            nn.Linear(dim, ff, bias=False),
            nn.SiLU(),
            nn.Linear(ff, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            attn_mask: (B, T) bool (1 = valid, 0 = pad) — pad 위치는 attention skip
        """
        B, T, D = x.shape
        residual = x
        h = self.norm1(x)

        uvqk = self.uvqk_proj(h).chunk(4, dim=-1)
        u, v, q, k = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in uvqk]
        u = F.silu(u)
        v = F.silu(v)

        # SDPA 는 is_causal 과 attn_mask 동시 사용 불가 → mask 결합
        if attn_mask is not None:
            # padding mask (B, T) → (B, 1, 1, T)
            pad = attn_mask[:, None, None, :].bool()
            if self.causal:
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                combined = pad & causal[None, None]
            else:
                combined = pad
            agg = F.scaled_dot_product_attention(
                q, k, v, attn_mask=combined,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            agg = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.causal,
            )

        # pointwise multiplication (gating from U)
        gated = agg * u
        gated = gated.transpose(1, 2).contiguous().view(B, T, D)
        gated = self.out_proj(gated)

        x = residual + self.dropout(gated)
        # second residual: gate FF
        x = x + self.dropout(self.gate(self.norm2(x)))
        return x


class HSTUEncoder(nn.Module):
    """N 개의 HSTUBlock 을 쌓은 sequence encoder."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HSTUBlock(dim=dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x)
