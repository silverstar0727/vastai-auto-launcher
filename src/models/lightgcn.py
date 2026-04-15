"""LightGCN 추천 모델 LightningModule.

기존 apps/lightgcn/models/lightgcn_model.py의 학습/평가 로직을
reco-lightning의 LightningModule 패턴으로 마이그레이션.

CBF/Complement와의 핵심 차이점:
1. Graph Convolution 기반 (Two-tower가 아님)
2. 유저/아이템 공유 임베딩 공간
3. 학습 시 full graph forward pass 필요
4. Learnable temperature loss (CE/BPR/BCE)
5. History fusion gate
"""

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import lightning as L
from torchmetrics import Metric
from torch import Tensor

from nets.lightgcn.lightgcn_net import LightGCNNet
from nets.lightgcn.losses import LOSS_REGISTRY, Loss
from nets.lightgcn.evaluator import AccuracyAndCoverage


# --- Metrics ---


class LossAccumulator(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, loss):
        self.loss_sum += loss.detach()
        self.total += 1

    def compute(self, reset=True):
        avg_loss = self.loss_sum / self.total
        if reset:
            self.reset()
        return avg_loss


# --- LightningModule ---


class LightGCNModel(L.LightningModule):
    def __init__(
        self,
        # 모델 아키텍처
        embedding_dim: int = 192,
        num_layers: int = 3,
        normalize: bool = True,
        # History fusion
        use_learned_gate: bool = True,
        history_pooling: str = "weighted_sum",
        # Loss
        loss_type: str = "ce",
        loss_temperature: float = 0.05,
        learnable_temperature: bool = True,
        temperature_min: float = 0.03,
        temperature_max: float = 0.1,
        bce_pos_weight: float = 0.25,
        # Negative sampling
        use_inbatch_neg: bool = True,
        logq_correction: bool = True,
        # Optimizer
        lr: float = 0.005,
        weight_decay: float = 0.01,
        # Metrics
        metric_ks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if metric_ks is None:
            self.hparams.metric_ks = [10, 20, 50]

        self.net = None
        self.graph_loader = None
        self.loss_fn = None
        self.train_loss_acc = None
        self.val_loss_acc = None
        self.val_metric = None
        self._val_user_emb: Optional[Tensor] = None
        self._val_item_emb: Optional[Tensor] = None
        self._val_gate_values: list[Tensor] = []
        self._epoch_start_time = 0.0

    def setup(self, stage):
        if self.net is not None:
            return

        dm = self.trainer.datamodule

        self.net = LightGCNNet(
            num_users=dm.num_users,
            num_items=dm.num_items,
            embedding_dim=self.hparams.embedding_dim,
            num_layers=self.hparams.num_layers,
            normalize=self.hparams.normalize,
            use_learned_gate=self.hparams.use_learned_gate,
            history_pooling=self.hparams.history_pooling,
            degree_inv_sqrt=dm.degree_inv_sqrt,
        )

        self.graph_loader = dm.train_graph_loader

        # log-Q correction용 item frequency
        if self.hparams.logq_correction and self.hparams.loss_type == "ce" and self.hparams.use_inbatch_neg:
            if dm.item_freq is not None:
                self.register_buffer("log_item_freq", dm.item_freq.clamp_min(1e-12).log())
            else:
                self.log_item_freq = None
        else:
            self.log_item_freq = None

        self.loss_fn = LOSS_REGISTRY[self.hparams.loss_type](
            temp=self.hparams.loss_temperature,
            pos_weight=self.hparams.bce_pos_weight,
            learnable_temp=self.hparams.learnable_temperature,
            temp_min=self.hparams.temperature_min,
            temp_max=self.hparams.temperature_max,
        )

        self.train_loss_acc = LossAccumulator()
        self.val_loss_acc = LossAccumulator()
        self.val_metric = AccuracyAndCoverage(dm.num_items, self.hparams.metric_ks)
        self.std_val_metric = AccuracyAndCoverage(dm.num_items, self.hparams.metric_ks)  # 논문 표준 메트릭

    def on_train_epoch_start(self):
        self._epoch_start_time = time.perf_counter()

    def training_step(self, batch: Dict[str, Tensor], batch_idx) -> Tensor:
        adj_mat = self.graph_loader.load(self.device)
        emb_all = self.net(adj_mat)
        user_emb_all, item_emb_all = self.net.split_user_item_emb(emb_all)

        u_emb = user_emb_all[batch["user_id"]]
        pos_i_emb = item_emb_all[batch["item_id"]]
        neg_i_emb = item_emb_all[batch["neg_item_ids"]]

        history_emb = self.net.pool_history(
            item_emb_all, batch["history_item_ids"], batch["history_mask"], batch["history_len"],
        )
        if self.hparams.use_learned_gate:
            u_emb, _ = self.net.fuse_with_learned_gate(u_emb, history_emb, batch["history_len"])
        else:
            u_emb = self.net.fuse_with_fixed_weight(u_emb, history_emb, batch["history_len"])

        loss = self._compute_loss(u_emb, pos_i_emb, neg_i_emb, batch["item_id"])
        self.train_loss_acc(loss.detach())
        return loss

    def on_validation_epoch_start(self):
        adj_mat = self.graph_loader.load(self.device)
        emb_all = self.net(adj_mat)
        self._val_user_emb, self._val_item_emb = self.net.split_user_item_emb(emb_all)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx):
        u_emb = self._val_user_emb[batch["user_id"]]
        pos_i_emb = self._val_item_emb[batch["item_id"]]
        neg_i_emb = self._val_item_emb[batch["neg_item_ids"]]

        history_emb = self.net.pool_history(
            self._val_item_emb, batch["history_item_ids"], batch["history_mask"], batch["history_len"],
        )
        if self.hparams.use_learned_gate:
            u_emb, g = self.net.fuse_with_learned_gate(u_emb, history_emb, batch["history_len"])
            has_history = batch["history_len"] > 0
            self._val_gate_values.append(g[has_history].detach())
        else:
            u_emb = self.net.fuse_with_fixed_weight(u_emb, history_emb, batch["history_len"])

        loss = self._compute_loss(u_emb, pos_i_emb, neg_i_emb, batch["item_id"])
        self.val_loss_acc(loss.detach())

        # full ranking metric (chunk 단위로 VRAM 절약)
        item_emb_t = self._val_item_emb.T
        target = batch["item_id"]
        hist_ids = batch["history_item_ids"]  # [B, S]
        chunk_size = 256
        for start in range(0, u_emb.size(0), chunk_size):
            end = start + chunk_size
            scores = torch.matmul(u_emb[start:end], item_emb_t)

            # 기존 메트릭 (히스토리 제거 없음)
            self.val_metric.update(scores, target[start:end])

            # 논문 표준 메트릭: full ranking + history removal
            std_scores = scores.clone()
            std_scores.scatter_(1, hist_ids[start:end], float("-inf"))
            self.std_val_metric.update(std_scores, target[start:end])

    def on_validation_epoch_end(self):
        self._val_user_emb = None
        self._val_item_emb = None

        val_loss = self.val_loss_acc.compute()
        self.log("val/loss", val_loss)
        train_loss = self.train_loss_acc.compute()
        self.log("train/loss", train_loss)

        metrics = self.val_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar="Recall" in k)

        std_metrics = self.std_val_metric.compute()
        for k, v in std_metrics.items():
            self.log(f"val/std_{k}", v)

        temp = self.loss_fn.temp.item()
        self.log("train/temp", temp)

        if self.hparams.use_learned_gate and self._val_gate_values:
            all_g = torch.cat(self._val_gate_values).squeeze(-1)
            self.log("val/gate_mean", all_g.mean())
            self._val_gate_values = []

        elapsed = time.perf_counter() - self._epoch_start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        self.print(
            f"[Epoch {self.current_epoch}] ({h}h {m:02d}m {s:02d}s) "
            f"train_loss={train_loss:.5f}; val_loss={val_loss:.5f}; temp={temp:.5f}"
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def _compute_loss(
        self, u_emb: Tensor, pos_i_emb: Tensor, neg_i_emb: Tensor, pos_item_ids: Tensor,
    ) -> Tensor:
        pos_scores = (u_emb * pos_i_emb).sum(dim=1)
        neg_scores = torch.bmm(neg_i_emb, u_emb.unsqueeze(2)).squeeze(2)
        inbatch_scores = u_emb @ pos_i_emb.T if self.hparams.use_inbatch_neg else None

        # log-Q correction (in-batch negative의 popularity bias 보정)
        if self.log_item_freq is not None and inbatch_scores is not None:
            log_q = self.log_item_freq[pos_item_ids]
            correction = self.loss_fn.temp * log_q
            pos_scores = pos_scores - correction
            inbatch_scores = inbatch_scores - correction.unsqueeze(0)

        if inbatch_scores is not None:
            mask = ~torch.eye(inbatch_scores.size(0), dtype=torch.bool, device=inbatch_scores.device)
            inbatch_neg = inbatch_scores[mask].view(inbatch_scores.size(0), -1)
            neg_scores = torch.cat([neg_scores, inbatch_neg], dim=1)

        return self.loss_fn(pos_scores, neg_scores)

    def configure_optimizers(self):
        temp_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        temp_ids = {id(p) for p in temp_params}
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in temp_ids]

        return torch.optim.AdamW([
            {"params": other_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            {"params": temp_params, "lr": self.hparams.lr * 0.1, "weight_decay": 0.0},
        ])

    @torch.inference_mode()
    def export_embeddings(self, graph_loader=None) -> tuple[Tensor, Tensor]:
        """학습 후 전체 임베딩을 추출한다. graph_loader를 지정하면 해당 그래프로 계산."""
        self.eval()
        loader = graph_loader or self.graph_loader
        adj_mat = loader.load(self.device)
        emb_all = self.net(adj_mat)
        user_emb, item_emb = self.net.split_user_item_emb(emb_all)
        return user_emb.detach().cpu(), item_emb.detach().cpu()
