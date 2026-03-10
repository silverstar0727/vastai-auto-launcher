import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from torchmetrics import Metric
from torchmetrics.functional import auroc

logger = logging.getLogger(__name__)


# --- Label code ---

NEUTRAL = -1  # Sample not applicable to this task


# --- Metrics ---


class TaskLossAccumulator(Metric):
    """Accumulates per-task weighted loss sums and sample counts."""

    full_state_update = False

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = task_names
        for name in task_names:
            self.add_state(f"{name}_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"n_{name}", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, losses: dict[str, torch.Tensor], counts: dict[str, torch.Tensor]):
        for name in self.task_names:
            getattr(self, f"{name}_loss").add_(losses[name].detach())
            getattr(self, f"n_{name}").add_(counts[name].detach())

    def compute(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.task_names:
            n = getattr(self, f"n_{name}")
            loss_sum = getattr(self, f"{name}_loss")
            result[f"{name}_loss"] = loss_sum / n.clamp(min=1)
        return result


class TaskAUCAccumulator(Metric):
    """Accumulates predictions and targets per task for AUC computation."""

    full_state_update = False

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = task_names
        for name in task_names:
            self.add_state(f"{name}_preds", default=[], dist_reduce_fx="cat")
            self.add_state(f"{name}_targets", default=[], dist_reduce_fx="cat")

    def update(self, task_preds: dict[str, torch.Tensor], task_targets: dict[str, torch.Tensor]):
        for name in self.task_names:
            if task_preds[name].numel() > 0:
                getattr(self, f"{name}_preds").append(task_preds[name].detach())
                getattr(self, f"{name}_targets").append(task_targets[name].detach())

    def compute(self) -> dict[str, torch.Tensor]:
        result = {}
        for name in self.task_names:
            preds_list = getattr(self, f"{name}_preds")
            targets_list = getattr(self, f"{name}_targets")
            if len(preds_list) == 0:
                result[f"{name}_auc"] = torch.tensor(0.0)
                continue
            preds = torch.cat(preds_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            if targets.unique().numel() < 2:
                result[f"{name}_auc"] = torch.tensor(0.0)
            else:
                result[f"{name}_auc"] = auroc(preds=preds, target=targets.long(), task="binary")
        return result


# --- Model ---


class RankerDinModel(L.LightningModule):
    """Multi-task ranking model with DIN attention pooling.

    Trains three base towers (CTR, click_to_action, action_to_order) and derives
    composite tasks (CTCAR, CTCVR, CVR) via probability chain rule.

    Supports:
        - Freezing MLP layers after the first epoch
        - Reducing dropout after the first epoch
        - Per-task loss weighting
        - LR scaling after the first epoch
    """

    # All tasks used for loss and logging
    ALL_TASK_NAMES = [
        "ctr", "click_to_action", "ctcar", "ctcvr", "action_to_order", "cvr",
    ]

    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.00001,
        lr_scale_after_one_epoch: float = 1.0,
        dropout: float = 0.3,
        dropout_after_one_epoch: float = 0.1,
        freeze_mlp_after_one_epoch: bool = True,
        task_weight_ctr: float = 1.0,
        task_weight_ctcar: float = 1.0,
        task_weight_ctcvr: float = 1.0,
        task_weight_click_to_action: float = 0.5,
        task_weight_action_to_order: float = 0.25,
        task_weight_cvr: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        # Task weights
        self.task_weights = {
            "ctr": task_weight_ctr,
            "click_to_action": task_weight_click_to_action,
            "ctcar": task_weight_ctcar,
            "ctcvr": task_weight_ctcvr,
            "action_to_order": task_weight_action_to_order,
            "cvr": task_weight_cvr,
        }

        # Metrics
        self.train_loss_acc = TaskLossAccumulator(self.ALL_TASK_NAMES)
        self.val_loss_acc = TaskLossAccumulator(self.ALL_TASK_NAMES)
        self.val_auc_acc = TaskAUCAccumulator(self.ALL_TASK_NAMES)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        click_query: torch.Tensor,
        click_keys: torch.Tensor,
        click_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(user_features, item_features, click_query, click_keys, click_mask)

    def _compute_task_probs(
        self,
        logit_ctr: torch.Tensor,
        logit_c2a: torch.Tensor,
        logit_a2o: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all task probabilities from the three base tower logits."""
        p_ctr = torch.sigmoid(logit_ctr)
        p_c2a = torch.sigmoid(logit_c2a)
        p_a2o = torch.sigmoid(logit_a2o)
        return {
            "ctr": p_ctr,
            "click_to_action": p_c2a,
            "action_to_order": p_a2o,
            "ctcar": p_ctr * p_c2a,
            "ctcvr": p_ctr * p_c2a * p_a2o,
            "cvr": p_c2a * p_a2o,
        }

    def _compute_losses(
        self,
        task_probs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute per-task BCE losses over valid (non-neutral) samples.

        Returns:
            losses: dict of summed loss per task.
            counts: dict of valid sample count per task.
        """
        losses = {}
        counts = {}
        for name in self.ALL_TASK_NAMES:
            targets = labels[name]
            valid = targets != NEUTRAL
            n_valid = valid.sum()
            if n_valid > 0:
                with torch.amp.autocast("cuda", enabled=False):
                    loss = F.binary_cross_entropy(
                        task_probs[name][valid].float().clamp(0, 1),
                        targets[valid].float().clamp(0, 1),
                        weight=weights[name][valid].float(),
                        reduction="sum",
                    )
            else:
                loss = torch.zeros(1, device=self.device)
            losses[name] = loss
            counts[name] = n_valid
        return losses, counts

    def _unpack_batch(self, batch):
        """Unpack a batch into model inputs and labels.

        Expected batch format:
            (user_features, item_features, click_query, click_keys, click_mask,
             labels_dict, weights_dict)
        where labels_dict and weights_dict have keys matching ALL_TASK_NAMES.
        """
        (
            user_features,
            item_features,
            click_query,
            click_keys,
            click_mask,
            labels,
            weights,
        ) = batch
        return user_features, item_features, click_query, click_keys, click_mask, labels, weights

    def training_step(self, batch, batch_idx):
        user_feat, item_feat, click_q, click_k, click_m, labels, weights = self._unpack_batch(batch)
        logit_ctr, logit_c2a, logit_a2o = self(user_feat, item_feat, click_q, click_k, click_m)

        task_probs = self._compute_task_probs(logit_ctr, logit_c2a, logit_a2o)
        losses, counts = self._compute_losses(task_probs, labels, weights)

        # Weighted total loss
        total_loss = torch.zeros(1, device=self.device)
        for name in self.ALL_TASK_NAMES:
            n = counts[name].clamp(min=1)
            total_loss = total_loss + self.task_weights[name] * losses[name] / n

        self.train_loss_acc.update(losses, counts)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_loss_acc.compute()
        for k, v in metrics.items():
            self.log(f"train/{k}", v)

        # After first epoch: freeze MLP and reduce dropout
        if self.current_epoch == 0:
            if self.hparams.freeze_mlp_after_one_epoch:
                logger.info("Freezing MLP layers after first epoch")
                freeze_targets = [
                    self.net.tower_cross_layers,
                    self.net.tower_hidden_layers,
                    self.net.out_layers,
                    self.net.din_pool,
                ]
                for module in freeze_targets:
                    for param in module.parameters():
                        param.requires_grad = False

            new_dropout = self.hparams.dropout_after_one_epoch
            logger.info(f"Reducing dropout to {new_dropout} after first epoch")
            self.net.set_dropout(new_dropout)

    def validation_step(self, batch, batch_idx):
        user_feat, item_feat, click_q, click_k, click_m, labels, weights = self._unpack_batch(batch)
        logit_ctr, logit_c2a, logit_a2o = self(user_feat, item_feat, click_q, click_k, click_m)

        task_probs = self._compute_task_probs(logit_ctr, logit_c2a, logit_a2o)
        losses, counts = self._compute_losses(task_probs, labels, weights)

        # Total val loss for checkpointing
        total_loss = torch.zeros(1, device=self.device)
        for name in self.ALL_TASK_NAMES:
            n = counts[name].clamp(min=1)
            total_loss = total_loss + self.task_weights[name] * losses[name] / n
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_loss_acc.update(losses, counts)

        # Collect preds/targets for AUC
        task_preds = {}
        task_targets = {}
        for name in self.ALL_TASK_NAMES:
            valid = labels[name] != NEUTRAL
            task_preds[name] = task_probs[name][valid].reshape(-1)
            task_targets[name] = labels[name][valid].long().reshape(-1)
        self.val_auc_acc.update(task_preds, task_targets)

    def on_validation_epoch_end(self) -> None:
        loss_metrics = self.val_loss_acc.compute()
        for k, v in loss_metrics.items():
            self.log(f"val/{k}", v)

        auc_metrics = self.val_auc_acc.compute()
        for k, v in auc_metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scale = self.hparams.lr_scale_after_one_epoch

        def lr_lambda(epoch):
            return 1.0 if epoch < 1 else scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
