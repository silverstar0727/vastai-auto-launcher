"""SIDv2 추가 손실 — Co-occurrence contrastive (PLUM Sec 4.3).

L_recon, L_vq 는 RQVAE.forward 에서 산출.
여기서는 contrastive (그리고 옵션 DAS multi-view alignment) 정의.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def co_occurrence_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch NT-Xent (SimCLR style) — anchor 가 positive 와 가까워지고 다른 배치 멤버와 멀어지게.

    Args:
        anchor:   (B, D) — quantized representation (RQ-VAE 의 q)
        positive: (B, D) — co-occurring item 의 quantized representation
        temperature: softmax 온도.

    PLUM:
        L_con = -Σ_i log exp(sim(p_i, p_i⁺)) / Σ_j exp(sim(p_i, p_j))
        (p_i⁺ = co-occurring item, p_j = batch 의 모든 item)
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.t() / temperature  # (B, B)
    labels = torch.arange(a.size(0), device=a.device)
    # symmetric: a→p, p→a
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss
