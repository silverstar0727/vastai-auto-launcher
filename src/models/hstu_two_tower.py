"""HSTU two-tower LightningModule.

학습: next-item prediction with multi-behavior weighting + in-batch sampled softmax.
평가: Recall@K, NDCG@K (전체 item vocab 대상 ranking; 시퀀스 마지막 hidden 사용).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Metric

from nets.hstu import HSTUTwoTowerNet


# behavior 별 학습 가중치 (PoC 기본; click 1, like 3, cart 5, purchase 20)
BEHAVIOR_WEIGHTS: Dict[int, float] = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 20.0}


class _MetricAcc(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, v):
        self.sum += v.detach()
        self.n += 1

    def compute(self):
        return self.sum / self.n.clamp(min=1)


def _recall_ndcg(scores: torch.Tensor, target: torch.Tensor, ks=(10, 50)) -> Dict[str, torch.Tensor]:
    """
    scores: (B, V) — V = num_items
    target: (B,)   — gold item id
    """
    # rank of target
    target_score = scores.gather(1, target.unsqueeze(1))  # (B,1)
    rank = (scores > target_score).sum(dim=1) + 1  # 1-indexed
    out = {}
    for k in ks:
        hit = (rank <= k).float()
        out[f"Recall@{k}"] = hit.mean()
        # NDCG = 1/log2(rank+1) if hit else 0
        ndcg = (hit * (1.0 / torch.log2(rank.float() + 1.0)))
        out[f"NDCG@{k}"] = ndcg.mean()
    return out


class HSTUTwoTowerModel(L.LightningModule):
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        num_interests: int = 4,
        attention_dim: int = 64,
        item_content_dim: int = 0,
        dropout: float = 0.1,
        # learning
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # eval ks
        eval_ks: tuple = (10, 50),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = None
        self.loss_acc = _MetricAcc()

    def setup(self, stage: str):
        if self.net is not None:
            return
        dm = self.trainer.datamodule
        self.net = HSTUTwoTowerNet(
            num_items=dm.num_items,
            num_behaviors=dm.num_behaviors,
            max_seq_len=dm.hparams.max_seq_len,
            dim=self.hparams.dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            ff_dim=self.hparams.ff_dim,
            num_interests=self.hparams.num_interests,
            attention_dim=self.hparams.attention_dim,
            item_content_dim=self.hparams.item_content_dim,
            dropout=self.hparams.dropout,
        )

    def _user_repr(self, batch):
        return self.net.encode_user(
            item_ids=batch["item_ids"],
            behavior_ids=batch["behavior_ids"],
            positions=batch["positions"],
            mask=batch["mask"],
            content_emb=batch.get("content_emb"),
        )  # (B, K, D)

    def training_step(self, batch, batch_idx):
        user_interests = self._user_repr(batch)  # (B, K, D)
        # in-batch sampled softmax 대신 full-vocab CE (active vocab 1.4M+1 정도)
        # full-vocab 는 메모리 부담 → 첫 PoC: in-batch negatives
        # target item embedding 만 추출
        target = batch["target_item"]                 # (B,)
        # item tower: 모든 batch target item 의 embedding (B, D)
        target_vec = self.net.item_emb(target)
        if self.net.content_proj is not None and "target_content" in batch:
            target_vec = target_vec + self.net.content_proj(batch["target_content"])

        # user max-over-K · target → (B, B) logits (in-batch)
        # u_i · t_j  for all i, j → (B, B)
        u_max = user_interests.max(dim=1).values     # (B, D)  — simplification
        logits = u_max @ target_vec.t()              # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        # 가중치: target_behavior 기반
        weights = torch.tensor(
            [BEHAVIOR_WEIGHTS.get(b.item(), 1.0) for b in batch["target_behavior"]],
            device=logits.device,
        )
        loss_per = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss_per * weights).sum() / weights.sum().clamp(min=1e-6)
        self.loss_acc(loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train/loss", self.loss_acc.compute(), prog_bar=True)
        self.loss_acc.reset()

    def validation_step(self, batch, batch_idx):
        user_interests = self._user_repr(batch)
        u_max = user_interests.max(dim=1).values
        # 전체 item embedding (메모리: V × D — 1.4M × 256 = 1.4GB fp32, fp16이면 0.7GB)
        # PoC: max 50K 후보 sampling 추천하지만 일단 full
        item_table = self.net.item_emb.weight       # (V, D)
        logits = u_max @ item_table.t()             # (B, V)
        # exclude pad item id 0
        logits[:, 0] = -1e9
        # 입력 시퀀스 안의 item 은 점수 깎기 (history 제외)
        logits.scatter_(1, batch["item_ids"], -1e9)
        target = batch["target_item"]
        metrics = _recall_ndcg(logits, target, ks=tuple(self.hparams.eval_ks))
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=("Recall@10" in k), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
