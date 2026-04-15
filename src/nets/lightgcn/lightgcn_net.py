"""LightGCN 네트워크 아키텍처.

기존 apps/lightgcn/models/lightgcn_model.py에서 순수 nn.Module 부분을 분리.

Graph Convolution으로 유저/아이템 임베딩을 학습하고,
유저 히스토리 pooling + ScalarGate를 통해 최종 유저 임베딩을 생성한다.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

TRAINED_EMB_FIXED_WEIGHT = 0.8


class SymmetricSpMM(torch.autograd.Function):
    """대칭 인접 행렬에 대한 SpMM (adj == adj.T 가정)."""

    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, grad_out):
        adj = ctx.adj
        grad_x = torch.sparse.mm(adj, grad_out)
        return None, grad_x


class LightConvolution(nn.Module):
    def forward(self, adj_mat: Tensor, x: Tensor) -> Tensor:
        return SymmetricSpMM.apply(adj_mat, x)


class ScalarGate(nn.Module):
    """GNN 유저 임베딩과 히스토리 pooling 임베딩을 학습된 gate로 퓨전."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.gate = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_emb: Tensor, recent_emb: Tensor, has_history: Tensor) -> tuple[Tensor, Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([user_emb, recent_emb], dim=-1)))
        fused = g * user_emb + (1 - g) * recent_emb
        fused_emb = torch.where(has_history.unsqueeze(-1), fused, user_emb)
        return fused_emb, g


class LightGCNNet(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        normalize: bool,
        use_learned_gate: bool,
        history_pooling: Literal["weighted_sum", "mean"],
        degree_inv_sqrt: Tensor,
    ):
        """
        Args:
            num_users: 유저 수 (unknown 제외)
            num_items: 아이템 수 (unknown 제외)
            embedding_dim: 임베딩 차원
            num_layers: GCN 레이어 수
            normalize: L2 정규화 여부
            use_learned_gate: 학습된 gate 사용 여부 (False이면 fixed weight)
            history_pooling: 히스토리 pooling 방식 ("weighted_sum" or "mean")
            degree_inv_sqrt: 아이템 degree의 역제곱근 (weighted_sum pooling에 사용)
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.normalize = normalize
        self.use_learned_gate = use_learned_gate
        self.history_pooling = history_pooling

        self.register_buffer("degree_inv_sqrt", degree_inv_sqrt)

        # 유저 + 아이템 공유 임베딩 공간
        # [0..num_users]: user (0=unknown), [num_users+1]: item padding, [num_users+2..]: items
        self.embedding = nn.Embedding(num_users + num_items + 2, embedding_dim, padding_idx=num_users + 1)
        self.conv_layers = nn.ModuleList([LightConvolution() for _ in range(num_layers)])

        if use_learned_gate:
            self.scalar_gate = ScalarGate(embedding_dim)

    def forward(self, adj_mat: Tensor) -> Tensor:
        """GCN forward. 전체 유저+아이템 임베딩을 리턴한다."""
        x = self.embedding.weight
        out = x
        for conv in self.conv_layers:
            x = conv(adj_mat, x)
            out = out + x
        out = out / (self.num_layers + 1)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=1)
        return out

    def split_user_item_emb(self, emb_all: Tensor) -> tuple[Tensor, Tensor]:
        """전체 임베딩을 user/item으로 분리."""
        return torch.split(emb_all, [self.num_users + 1, self.num_items + 1])

    def pool_history(
        self,
        item_emb_all: Tensor,
        history_item_ids: Tensor,
        history_mask: Tensor,
        history_len: Tensor,
    ) -> Tensor:
        """유저 히스토리 아이템 임베딩을 pooling."""
        history_emb = item_emb_all[history_item_ids]  # (B, S, D)
        history_emb = history_emb * history_mask.unsqueeze(-1).float()
        if self.history_pooling == "weighted_sum":
            weights = self.degree_inv_sqrt[history_item_ids] * history_mask.float()
            pooled_emb = torch.bmm(weights.unsqueeze(1), history_emb).squeeze(1)
        else:
            pooled_emb = history_emb.sum(dim=1) / history_len.clamp(min=1).float().unsqueeze(-1)
        if self.normalize:
            pooled_emb = F.normalize(pooled_emb, p=2.0, dim=1)
        return pooled_emb

    def fuse_with_learned_gate(
        self, u_emb: Tensor, history_emb: Tensor, history_len: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """학습된 gate로 GNN 임베딩과 히스토리 임베딩을 퓨전."""
        has_history = history_len > 0
        fused_emb, g = self.scalar_gate(u_emb, history_emb, has_history)
        if self.normalize:
            fused_emb = F.normalize(fused_emb, p=2.0, dim=1)
        return fused_emb, g

    def fuse_with_fixed_weight(
        self, u_emb: Tensor, history_emb: Tensor, history_len: Tensor,
    ) -> Tensor:
        """고정 가중치로 GNN 임베딩과 히스토리 임베딩을 퓨전."""
        has_history = history_len > 0
        fused_emb = TRAINED_EMB_FIXED_WEIGHT * u_emb + (1 - TRAINED_EMB_FIXED_WEIGHT) * history_emb
        fused_emb = torch.where(has_history.unsqueeze(-1), fused_emb, u_emb)
        if self.normalize:
            fused_emb = F.normalize(fused_emb, p=2.0, dim=1)
        return fused_emb
