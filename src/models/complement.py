from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

import lightning as L
from torchmetrics import Metric

from nets.complement.two_tower import ComplementNet
from utils.accuracy import recalls_and_ndcgs_for_ks
from utils.constants import FeatureField


# --- Metrics (CBFModel과 동일) ---


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


class ComplementModel(L.LightningModule):
    def __init__(
        self,
        top_k: int = 50,
        negative_sampling: bool = False,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        item_dropout_prob: float = 0.0,
        optimizer_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = None
        self.accuracy_metric = None
        self.coverage_metric = None
        self.loss_acc = None

    def setup(self, stage):
        if self.net is not None:
            return

        dm = self.trainer.datamodule

        self.net = ComplementNet(
            num_items=dm.num_items,
            features=dm.features,
            user_tower_features=dm.user_feat_names,
            item_tower_features=dm.item_feat_names,
            item_feat_values=dm.item_feat_values,
            num_hidden_layers=self.hparams.num_hidden_layers,
            last_hidden_units=self.hparams.last_hidden_units,
            item_dropout_prob=self.hparams.item_dropout_prob,
        )

        self.accuracy_metric = Accuracy(self.hparams.top_k)
        self.coverage_metric = Coverage(dm.num_items, self.hparams.top_k)
        self.loss_acc = LossAccumulator()

    def training_step(self, batch, batch_idx):
        if self.hparams.negative_sampling:
            inputs, batch_pos_labels = batch
            in_batch_pos_items = torch.unique(batch_pos_labels.reshape(-1,))

            pred_scores = self.net(inputs, in_batch_pos_items)
            target_scores = create_target_scores(pred_scores, batch_pos_labels, in_batch_pos_items)

            loss = -torch.mean(torch.sum(F.log_softmax(pred_scores, 1) * target_scores, -1))
        else:
            inputs, labels = batch
            scores = self.net(inputs)

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
                weight_decay=params.get("weight_decay", 1e-3),
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        lr_scheduler_name = params.get("lr_scheduler", None)
        if lr_scheduler_name is None:
            return optimizer

        if lr_scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=params.get("lr_scheduler_step", 10),
                gamma=params.get("lr_scheduler_gamma", 0.1),
            )
        else:
            raise ValueError(f"Invalid lr_scheduler name: {lr_scheduler_name}")
        return [optimizer], [scheduler]
