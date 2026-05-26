"""Multi-interest head — ComiRec-style.

마지막 hidden 또는 마지막 K 시점 hidden 으로부터 K 개 interest 벡터 추출.
PoC: self-attention 기반 (ComiRec-SA) — 학습 가능한 query K 개.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInterestHead(nn.Module):
    """
    Args:
        dim:               hidden 차원
        num_interests:     interest 개수 K
        attention_dim:     attention 내 dim (보통 dim//2)
    """

    def __init__(self, dim: int, num_interests: int = 4, attention_dim: int = 64):
        super().__init__()
        self.num_interests = num_interests
        self.W1 = nn.Linear(dim, attention_dim, bias=False)
        self.W2 = nn.Linear(attention_dim, num_interests, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) sequence hidden
            mask: (B, T) bool (1 = valid)
        Returns:
            (B, K, D) interest 벡터 K 개
        """
        att = self.W2(self.tanh(self.W1(h)))  # (B, T, K)
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        att = att.softmax(dim=1)               # softmax over T
        # weighted sum: (B, K, D)
        interests = att.transpose(1, 2) @ h
        return interests
