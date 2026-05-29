"""RQ-VAE LETTER variant (Wang et al., CIKM 2024, arXiv:2405.07314).

기존 `rq_vae.py` 의 RQVAE 를 상속하여 두 가지 정규화 loss 추가:
  - L_CF (collaborative regularization): sum-of-codeword embedding 을 frozen CF
    teacher embedding (SASRec / HSTU / LightGCN 등) 과 InfoNCE 로 align.
  - L_Div (diversity regularization): 매 N step 마다 codebook embedding 을
    constrained K-means 로 클러스터링 → intra-cluster pull, inter-cluster push.

기존 RQ-VAE 학습 경로에 영향 없음 — 본 모듈을 import 하지 않으면 기존 동작 그대로.
관련 doc: data/reco_experiments/v2_full/EXPERIMENT_MULTI_INTEREST_VS_TIGER.md (후속)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rq_vae import RQVAE


class RQVAELetter(RQVAE):
    """RQ-VAE + LETTER L_CF + L_Div.

    Args (RQVAE 상속 인자 외 신규):
        cf_emb_dim: CF teacher embedding 차원 (e.g., SASRec 32, HSTU 64-256).
                    latent_dim 과 다르면 projection Linear 자동 삽입.
        letter_div_clusters: L_Div constrained K-means 클러스터 수 (paper default K=10).
        letter_div_recluster_every: K-means 재실행 주기 (paper 미명시, 권장 500-1000 step).
    """

    def __init__(
        self,
        input_dim: int,
        cf_emb_dim: int = 64,
        letter_div_clusters: int = 10,
        letter_div_recluster_every: int = 500,
        **kwargs,
    ):
        super().__init__(input_dim=input_dim, **kwargs)
        latent_dim = self.quantizer.dim
        self.cf_emb_dim = cf_emb_dim
        if cf_emb_dim != latent_dim:
            self.cf_proj = nn.Linear(cf_emb_dim, latent_dim, bias=False)
        else:
            self.cf_proj = nn.Identity()

        self.letter_div_clusters = letter_div_clusters
        self.letter_div_recluster_every = letter_div_recluster_every

        self.register_buffer("_letter_step", torch.zeros(1, dtype=torch.long))
        for i, n in enumerate(self.quantizer.codebook_sizes):
            self.register_buffer(
                f"_letter_cluster_{i}", torch.zeros(n, dtype=torch.long)
            )
            self.register_buffer(
                f"_letter_cluster_ready_{i}", torch.zeros(1, dtype=torch.bool)
            )

    # -------- L_CF --------
    def compute_cf_loss(self, z_q: torch.Tensor, cf_emb: torch.Tensor) -> torch.Tensor:
        """LETTER Eq.3 — InfoNCE over batch with sum-of-codewords ẑ_i.

        z_q: (B, latent_dim) — RQ-VAE 의 quantized straight-through 출력
             (이미 모든 level codeword 합 = ẑ_i).
        cf_emb: (B, cf_emb_dim) — frozen CF teacher embedding.
        """
        cf = self.cf_proj(cf_emb)  # (B, latent_dim)
        # paper: ⟨ẑ_i, h_j⟩ 내적, in-batch denominator
        logits = z_q @ cf.t()  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    # -------- L_Div --------
    @torch.no_grad()
    def _recluster(self):
        from sklearn.cluster import KMeans

        for level in range(self.quantizer.num_levels):
            codebook = getattr(self.quantizer, f"codebook_{level}")
            data = codebook.float().cpu().numpy()
            n_clusters = min(self.letter_div_clusters, data.shape[0])
            km = KMeans(
                n_clusters=n_clusters, n_init=3, random_state=42, max_iter=50
            ).fit(data)
            labels = torch.from_numpy(km.labels_).long().to(codebook.device)
            getattr(self, f"_letter_cluster_{level}").copy_(labels)
            getattr(self, f"_letter_cluster_ready_{level}").fill_(True)

    def compute_div_loss(self) -> torch.Tensor:
        """LETTER Eq.4 — per-codebook intra-cluster pull, inter-cluster push.

        매 forward 호출 시 step 카운터 증가, recluster_every step 마다 K-means 재실행.
        positive = 같은 클러스터 내 random code, negatives = 그 외 모든 codebook code.
        """
        if self.training:
            step = int(self._letter_step.item())
            if step % self.letter_div_recluster_every == 0:
                self._recluster()
            self._letter_step += 1

        total_loss = 0.0
        n_levels = 0
        for level in range(self.quantizer.num_levels):
            ready = getattr(self, f"_letter_cluster_ready_{level}")
            if not bool(ready.item()):
                continue
            codebook = getattr(self.quantizer, f"codebook_{level}")  # (N, d)
            labels = getattr(self, f"_letter_cluster_{level}")  # (N,)
            N = codebook.size(0)

            # same-cluster mask, exclude self
            same = labels.unsqueeze(0) == labels.unsqueeze(1)
            same.fill_diagonal_(False)
            has_peer = same.any(dim=1)
            if not has_peer.any():
                continue

            valid_idx = torch.nonzero(has_peer, as_tuple=False).squeeze(1)
            valid_codes = codebook[valid_idx]  # (M, d)

            # random positive per row from same cluster
            rand = torch.rand(valid_idx.size(0), N, device=codebook.device)
            rand = rand.masked_fill(~same[valid_idx], -1.0)
            pos_idx = rand.argmax(dim=1)  # (M,)
            positives = codebook[pos_idx]

            # logits over all codes (excluding self)
            logits = valid_codes @ codebook.t()  # (M, N)
            logits[torch.arange(valid_idx.size(0), device=codebook.device), valid_idx] = float("-inf")
            denom = torch.logsumexp(logits, dim=1)
            pos_score = (valid_codes * positives).sum(dim=1)
            loss = -(pos_score - denom).mean()
            total_loss = total_loss + loss
            n_levels += 1

        if n_levels == 0:
            return self.quantizer.codebook_0.new_zeros(())
        return total_loss / n_levels
