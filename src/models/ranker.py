import torch
import torch.nn as nn

import lightning as L
from torchmetrics.classification import BinaryAUROC

# Label constants
NEUTRAL = -1  # ignore in loss computation
NEGATIVE = 0
POSITIVE = 1

TASK_NAMES = ["ctr", "click_to_action", "action_to_order", "ctcar", "ctcvr", "cvr"]
BASE_TASK_NAMES = ["ctr", "click_to_action", "action_to_order"]


class RankerModel(L.LightningModule):
    """Multi-task ranker LightningModule.

    Wraps RankerNet and handles:
    - Multi-task BCE loss with neutral label filtering
    - AUROC metrics per task (6 tasks: 3 base + 3 derived)
    - Post-epoch MLP freezing and dropout adjustment
    - LR scheduling with optional scale after first epoch
    """

    def __init__(
        self,
        net: nn.Module,
        task_weight_ctr: float = 1.0,
        task_weight_ctcar: float = 1.0,
        task_weight_ctcvr: float = 1.0,
        task_weight_click_to_action: float = 0.5,
        task_weight_action_to_order: float = 0.25,
        task_weight_cvr: float = 0.5,
        lr: float = 0.00001,
        lr_scale_after_one_epoch: float = 1.0,
        dropout_after_one_epoch: float = 0.3,
        freeze_mlp_after_one_epoch: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        self.task_weights = {
            "ctr": task_weight_ctr,
            "click_to_action": task_weight_click_to_action,
            "action_to_order": task_weight_action_to_order,
            "ctcar": task_weight_ctcar,
            "ctcvr": task_weight_ctcvr,
            "cvr": task_weight_cvr,
        }

        self.lr = lr
        self.lr_scale_after_one_epoch = lr_scale_after_one_epoch
        self.dropout_after_one_epoch = dropout_after_one_epoch
        self.freeze_mlp_after_one_epoch = freeze_mlp_after_one_epoch

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Metrics: AUROC per task for val and test
        for stage in ["val", "test"]:
            for task in TASK_NAMES:
                setattr(self, f"{stage}_auroc_{task}", BinaryAUROC())

    def forward(self, user_features, item_features):
        return self.net(user_features, item_features)

    def training_step(self, batch, batch_idx):
        user_features, item_features, labels, weights = batch
        logit_ctr, logit_c2a, logit_a2o = self.net(user_features, item_features)

        loss = self._compute_loss(logit_ctr, logit_c2a, logit_a2o, labels, weights)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_features, item_features, labels, weights = batch
        logit_ctr, logit_c2a, logit_a2o = self.net(user_features, item_features)

        loss = self._compute_loss(logit_ctr, logit_c2a, logit_a2o, labels, weights)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        probs = self._compute_probs(logit_ctr, logit_c2a, logit_a2o)
        self._update_metrics("val", probs, labels)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def test_step(self, batch, batch_idx):
        user_features, item_features, labels, weights = batch
        logit_ctr, logit_c2a, logit_a2o = self.net(user_features, item_features)

        loss = self._compute_loss(logit_ctr, logit_c2a, logit_a2o, labels, weights)
        self.log("test/loss", loss, sync_dist=True)

        probs = self._compute_probs(logit_ctr, logit_c2a, logit_a2o)
        self._update_metrics("test", probs, labels)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            if self.freeze_mlp_after_one_epoch:
                self.net.freeze_mlp()
            self.net.set_dropout(self.dropout_after_one_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scale = self.lr_scale_after_one_epoch

        def lr_lambda(epoch):
            if epoch == 0:
                return 1.0
            return scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # --- Internal helpers ---

    def _compute_loss(self, logit_ctr, logit_c2a, logit_a2o, labels, weights):
        """Compute weighted sum of per-task BCE losses, ignoring NEUTRAL labels."""
        # Derive logits for composite tasks
        p_ctr = torch.sigmoid(logit_ctr)
        p_c2a = torch.sigmoid(logit_c2a)
        p_a2o = torch.sigmoid(logit_a2o)

        # For composite tasks, we compute BCE on derived probabilities
        # We need logits for BCEWithLogitsLoss on base tasks,
        # and raw BCE on derived tasks
        task_logits = {
            "ctr": logit_ctr,
            "click_to_action": logit_c2a,
            "action_to_order": logit_a2o,
        }

        task_probs = {
            "ctcar": p_ctr * p_c2a,
            "ctcvr": p_ctr * p_c2a * p_a2o,
            "cvr": p_c2a * p_a2o,
        }

        total_loss = torch.tensor(0.0, device=logit_ctr.device)

        # Base tasks: use BCEWithLogitsLoss
        for task_name, logit in task_logits.items():
            label = labels[task_name]
            weight = weights.get(task_name, None)
            mask = label != NEUTRAL
            if mask.sum() == 0:
                continue

            task_loss = self.bce(logit[mask], label[mask].float())
            if weight is not None:
                task_loss = task_loss * weight[mask]
            task_loss = task_loss.sum() / mask.sum()
            total_loss = total_loss + self.task_weights[task_name] * task_loss

        # Derived tasks: use binary_cross_entropy on probabilities
        bce_fn = nn.functional.binary_cross_entropy
        for task_name, prob in task_probs.items():
            label = labels[task_name]
            weight = weights.get(task_name, None)
            mask = label != NEUTRAL
            if mask.sum() == 0:
                continue

            prob_masked = prob[mask].clamp(1e-7, 1 - 1e-7)
            task_loss = bce_fn(prob_masked, label[mask].float(), reduction="none")
            if weight is not None:
                task_loss = task_loss * weight[mask]
            task_loss = task_loss.sum() / mask.sum()
            total_loss = total_loss + self.task_weights[task_name] * task_loss

        return total_loss

    @staticmethod
    def _compute_probs(logit_ctr, logit_c2a, logit_a2o):
        """Compute all 6 task probabilities from the 3 base logits."""
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

    def _update_metrics(self, stage: str, probs: dict, labels: dict):
        """Update AUROC metrics for non-neutral samples per task."""
        for task in TASK_NAMES:
            label = labels[task]
            mask = label != NEUTRAL
            if mask.sum() == 0:
                continue
            metric = getattr(self, f"{stage}_auroc_{task}")
            metric.update(probs[task][mask], label[mask].long())

    def _log_metrics(self, stage: str):
        """Compute and log AUROC for all tasks."""
        for task in TASK_NAMES:
            metric = getattr(self, f"{stage}_auroc_{task}")
            try:
                value = metric.compute()
                self.log(f"{stage}/auroc_{task}", value, prog_bar=(task == "ctr"))
            except ValueError:
                pass  # Skip if no samples were accumulated
            metric.reset()
