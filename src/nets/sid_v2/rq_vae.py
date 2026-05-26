"""Residual-Quantized VAE (PLUM SIDv2 lineage).

Components:
  - Encoder/decoder MLP
  - ResidualQuantizer with **multi-resolution codebook**
    (codebook size shrinks by /2 per level — PLUM: 2048, 1024, 512, ...)
  - EMA codebook update (VQ-VAE 표준) 또는 straight-through estimator

Forward:
    z         = encoder(x)                       # (B, D_latent)
    q, codes  = quantizer(z)                     # q: (B, D), codes: (B, L)
    x_recon   = decoder(q)
    return {'recon': x_recon, 'codes': codes, 'z': z, 'q': q, 'vq_loss': ...}

References:
  - TIGER (Rajput et al., NeurIPS 2023): RQ-VAE for Semantic IDs
  - PLUM (Google, 2025, arXiv 2510.07784): SIDv2 with multi-resolution + co-occurrence
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    """Multi-level residual vector quantization with multi-resolution codebooks.

    Args:
        dim: latent dimension (encoder 출력 D).
        codebook_sizes: 각 레벨 codebook 크기. e.g., (2048, 1024, 512, 256).
            len(codebook_sizes) == 양자화 레벨 수.
        commitment_cost: VQ-VAE commitment loss 가중 (Oord et al. 2017 기본 0.25).
        ema_decay: 0 이면 standard VQ (straight-through), >0 이면 EMA codebook update.
            EMA 가 표준이고 codebook collapse 에 더 강건.
        eps: EMA 분모 안정화.
    """

    def __init__(
        self,
        dim: int,
        codebook_sizes: Tuple[int, ...] = (2048, 1024, 512, 256),
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_sizes = tuple(codebook_sizes)
        self.num_levels = len(codebook_sizes)
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.eps = eps

        # 각 레벨별 codebook (uniform init, scaled)
        for i, n in enumerate(codebook_sizes):
            embed = torch.randn(n, dim) * (1.0 / (dim ** 0.5))
            self.register_buffer(f"codebook_{i}", embed)
            self.register_buffer(f"ema_cluster_size_{i}", torch.zeros(n))
            self.register_buffer(f"ema_embed_{i}", embed.clone())

    def _quantize_level(self, residual: torch.Tensor, level: int):
        """현재 residual 을 level codebook 의 가장 가까운 entry 로 양자화."""
        codebook = getattr(self, f"codebook_{level}")
        # 거리 계산: (B, N) = ||r||² - 2 r·e + ||e||²
        dists = (
            residual.pow(2).sum(-1, keepdim=True)
            - 2 * residual @ codebook.t()
            + codebook.pow(2).sum(-1)
        )
        codes = dists.argmin(dim=-1)  # (B,)
        quantized = F.embedding(codes, codebook)  # (B, D)
        return quantized, codes, codebook

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, dim) encoder 출력.
        Returns:
            quantized: (B, dim) — sum of all levels (straight-through)
            codes: (B, num_levels) — 각 레벨의 codebook index
            vq_loss: scalar — commitment + (EMA 가 아니면) codebook loss
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_codes: List[torch.Tensor] = []
        commitment_loss = z.new_zeros(())
        codebook_loss = z.new_zeros(())

        for level in range(self.num_levels):
            q, codes, codebook = self._quantize_level(residual, level)
            all_codes.append(codes)
            quantized_sum = quantized_sum + q
            # commitment: encoder side
            commitment_loss = commitment_loss + F.mse_loss(residual, q.detach())
            if self.ema_decay <= 0:
                codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
            else:
                if self.training:
                    self._ema_update(level, residual.detach(), codes)
            residual = residual - q.detach()

        # straight-through
        quantized_st = z + (quantized_sum - z).detach()
        codes_out = torch.stack(all_codes, dim=-1)  # (B, L)
        vq_loss = self.commitment_cost * commitment_loss + codebook_loss
        return quantized_st, codes_out, vq_loss

    @torch.no_grad()
    def _ema_update(self, level: int, flat_input: torch.Tensor, codes: torch.Tensor):
        n = self.codebook_sizes[level]
        one_hot = F.one_hot(codes, n).type(flat_input.dtype)  # (B, N)
        new_cluster_size = one_hot.sum(0)
        new_embed_sum = one_hot.t() @ flat_input  # (N, D)

        ema_cluster_size = getattr(self, f"ema_cluster_size_{level}")
        ema_embed = getattr(self, f"ema_embed_{level}")
        ema_cluster_size.mul_(self.ema_decay).add_(new_cluster_size, alpha=1 - self.ema_decay)
        ema_embed.mul_(self.ema_decay).add_(new_embed_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing
        total = ema_cluster_size.sum()
        smoothed = (ema_cluster_size + self.eps) / (total + n * self.eps) * total
        new_codebook = ema_embed / smoothed.unsqueeze(1)
        getattr(self, f"codebook_{level}").copy_(new_codebook)

    @torch.no_grad()
    def code_utilization(self) -> List[float]:
        """각 레벨에서 effective code 수 (entropy 기반). 균등할수록 1.0."""
        out = []
        for level in range(self.num_levels):
            cs = getattr(self, f"ema_cluster_size_{level}")
            p = cs / (cs.sum() + 1e-9)
            p = p[p > 0]
            ent = -(p * p.log()).sum()
            eff = ent.exp().item() / self.codebook_sizes[level]
            out.append(eff)
        return out


class RQVAE(nn.Module):
    """Residual-Quantized VAE wrapper (encoder MLP + ResidualQuantizer + decoder MLP).

    Args:
        input_dim: 입력 콘텐츠 임베딩 차원 (e.g., 512 distiluse 또는 fused multimodal).
        latent_dim: 양자화 latent 차원.
        codebook_sizes: ResidualQuantizer 와 동일.
        encoder_hidden: encoder MLP hidden 크기 리스트.
        decoder_hidden: decoder MLP hidden 크기 리스트.
        dropout: MLP dropout.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        codebook_sizes: Tuple[int, ...] = (2048, 1024, 512, 256),
        encoder_hidden: Tuple[int, ...] = (512, 256),
        decoder_hidden: Tuple[int, ...] = (256, 512),
        dropout: float = 0.0,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = _mlp(input_dim, encoder_hidden, latent_dim, dropout)
        self.decoder = _mlp(latent_dim, decoder_hidden, input_dim, dropout)
        self.quantizer = ResidualQuantizer(
            dim=latent_dim,
            codebook_sizes=codebook_sizes,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        q, codes, vq_loss = self.quantizer(z)
        x_recon = self.decoder(q)
        return {
            "x_recon": x_recon,
            "z": z,
            "q": q,
            "codes": codes,           # (B, L) 정수 인덱스 = Semantic ID
            "vq_loss": vq_loss,
        }

    @torch.no_grad()
    def encode_to_sid(self, x: torch.Tensor) -> torch.Tensor:
        """입력 임베딩 → Semantic ID tuple. (인덱싱·서빙 시 사용)"""
        z = self.encoder(x)
        _, codes, _ = self.quantizer(z)
        return codes


def _mlp(in_dim: int, hiddens: Tuple[int, ...], out_dim: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hiddens:
        layers += [nn.Linear(prev, h), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
