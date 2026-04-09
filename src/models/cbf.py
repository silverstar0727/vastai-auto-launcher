from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from torchmetrics import Metric

from nets.cbf.two_tower import CBFNet
from utils.accuracy import recalls_and_ndcgs_for_ks
from utils.constants import FeatureField
from utils.seed import fix_random_seed_as


# --- Metrics (from pl_metrics.py + pl_callbacks.py) ---


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


class Accuracy(Metric):
    def __init__(self, top_k=50, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0))
        self.add_state("NDCG", default=torch.tensor(0.0))
        self.add_state("Recall", default=torch.tensor(0.0))
        self.top_k = top_k
        self.compute_on_step = False

    def update(self, batch_scores: torch.Tensor, batch_positive_items: torch.Tensor):
        batch_rank_items = batch_scores.argsort(dim=1, descending=True)[:, : self.top_k]
        metrics = recalls_and_ndcgs_for_ks(batch_rank_items, batch_positive_items, [self.top_k])
        self.NDCG += metrics[f"NDCG_{self.top_k}"]
        self.Recall += metrics[f"Recall_{self.top_k}"]
        self.total += 1

    def compute(self):
        scores = {
            f"NDCG_{self.top_k}": self.NDCG / self.total,
            f"Recall_{self.top_k}": self.Recall / self.total,
        }
        self.cover_items = set()
        self.reset()
        return scores


class Coverage(Metric):
    def __init__(self, num_items, top_k=50, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.num_items = num_items
        self.cover_items = set()
        self.compute_on_step = False

    def update(self, batch_scores: torch.Tensor):
        batch_rank_items = batch_scores.argsort(dim=1, descending=True)[:, : self.top_k]
        ret_items = set(batch_rank_items.cpu().numpy().reshape(-1,))
        self.cover_items = self.cover_items.union(ret_items)

    def compute(self):
        scores = {
            f"Coverage_Percentage_{self.top_k}": len(self.cover_items) / self.num_items,
            f"Coverage_Counts_{self.top_k}": len(self.cover_items),
        }
        self.cover_items = set()
        self.reset()
        return scores


# --- Helper function ---


def create_target_scores(pred_scores, batch_pos_labels, in_batch_pos_items):
    target_scores = torch.zeros_like(pred_scores)

    for i, pos_labels in enumerate(batch_pos_labels):
        pos_items = pos_labels[pos_labels > 0]
        _, pos_items_in_target_scores = (pos_items.reshape(-1, 1) == in_batch_pos_items.reshape(1, -1)).nonzero(
            as_tuple=True
        )
        target_scores[i, pos_items_in_target_scores] = 1.0
    return target_scores


# --- LightningModule ---


class CBFModel(L.LightningModule):
    def __init__(
        self,
        top_k: int = 50,
        negative_sampling: bool = True,
        softmax_temperature: float = 1.0,
        bias_correction: bool = False,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        model_init_seed: int = 0,
        item_dropout_prob: float = 0.0,
        normalize_outputs: bool = False,
        optimizer_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # net은 setup()에서 DataModule의 전처리 결과를 읽어 초기화
        self.net = None
        self.accuracy_metric = None
        self.coverage_metric = None
        self.loss_acc = None

    def setup(self, stage):
        if self.net is not None:
            return

        dm = self.trainer.datamodule

        # 원본과 동일하게 모델 초기화 직전에 seed 설정
        fix_random_seed_as(self.hparams.model_init_seed)

        self.net = CBFNet(
            num_items=dm.num_items,
            features=dm.features,
            user_tower_features=dm.user_feat_names,
            item_tower_features=dm.item_feat_names,
            item_feat_values=dm.item_feat_values,
            num_hidden_layers=self.hparams.num_hidden_layers,
            last_hidden_units=self.hparams.last_hidden_units,
            item_dropout_prob=self.hparams.item_dropout_prob,
            normalize_outputs=self.hparams.normalize_outputs,
        )

        self.accuracy_metric = Accuracy(self.hparams.top_k)
        self.coverage_metric = Coverage(dm.num_items, self.hparams.top_k)
        self.loss_acc = LossAccumulator()

    def training_step(self, batch, batch_idx):
        if self.hparams.negative_sampling:
            if self.hparams.bias_correction:
                inputs, batch_pos_labels, batch_pos_probs = batch
                non_zero_idxes = batch_pos_labels > 0
                in_batch_pos_items = torch.unique(batch_pos_labels[non_zero_idxes])

                in_batch_pos_probs = torch.zeros_like(in_batch_pos_items).float()
                _, pos_idxes = (batch_pos_labels[non_zero_idxes][:, None] == in_batch_pos_items).nonzero(as_tuple=True)
                in_batch_pos_probs[pos_idxes] = batch_pos_probs[non_zero_idxes]
            else:
                inputs, batch_pos_labels = batch
                in_batch_pos_items = torch.unique(batch_pos_labels.reshape(-1))
                in_batch_pos_probs = None

            pred_scores = self.net(inputs, in_batch_pos_items)
            target_scores = create_target_scores(pred_scores, batch_pos_labels, in_batch_pos_items)

            pred_scores = pred_scores / self.hparams.softmax_temperature
            if self.hparams.bias_correction:
                pred_scores = pred_scores - in_batch_pos_probs.log().unsqueeze(0)
            loss = -torch.mean(torch.sum(F.log_softmax(pred_scores, 1) * target_scores, -1))
        else:
            inputs, labels = batch
            scores = self.net(inputs)
            scores = scores / self.hparams.softmax_temperature

            logits_ignore_unknown_index = scores[:, 1:]
            targets_ignore_unknown_index = labels[:, 1:]
            loss = -torch.mean(
                torch.sum(F.log_softmax(logits_ignore_unknown_index, 1) * targets_ignore_unknown_index, -1)
            )
        self.loss_acc(loss)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, val_batch, batch_idx):
        inputs, batch_pos_labels = val_batch

        if self.hparams.negative_sampling:
            if self.hparams.bias_correction:
                in_batch_items = torch.unique(batch_pos_labels[batch_pos_labels > 0])
            else:
                in_batch_items = torch.unique(batch_pos_labels.reshape(-1,))
            pred_scores = self.net(inputs, in_batch_items)

            targets = []
            for i, pos_labels in enumerate(batch_pos_labels):
                pos_items = pos_labels[pos_labels > 0]
                _, pos_loc_in_batch_items = (pos_items.reshape(-1, 1) == in_batch_items.reshape(1, -1)).nonzero(
                    as_tuple=True
                )
                targets.append(pos_loc_in_batch_items.reshape(1, -1))
            targets = torch.cat(targets, dim=0)

        else:
            inputs, targets = val_batch
            pred_scores = self.net(inputs)

        self.accuracy_metric.update(pred_scores, targets)

    def on_validation_epoch_end(self):
        dict_ = self.accuracy_metric.compute()
        for k, v in dict_.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        params = self.hparams.optimizer_params or {}
        optimizer_name = params.get("optimizer", "adam").lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get('weight_decay', 1e-3),
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=params.get('lr', 1e-3),
                weight_decay=params.get('weight_decay', 0.0),
            )
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        lr_scheduler_name = params.get('lr_scheduler', None)
        if lr_scheduler_name is None:
            return optimizer

        if lr_scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=params.get('lr_scheduler_step', 10),
                gamma=params.get('lr_scheduler_gamma', 0.1),
            )
        else:
            raise ValueError(f"Invalid lr_scheduler name: {lr_scheduler_name}")
        return [optimizer], [scheduler]
