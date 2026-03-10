import torch
import torch.nn as nn

import lightning as L
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class LossAccumulator(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0))
        self.add_state("loss", default=torch.tensor(0.0))

    def update(self, loss: torch.Tensor):
        self.loss += loss.detach()
        self.total += 1

    def compute(self):
        return self.loss / self.total


class ComplementaryV2Model(L.LightningModule):
    """Lightning wrapper for DCNv2 complementary product prediction.

    Metrics: PR-AUC, F1, Accuracy, Precision, Recall
    Loss: BCEWithLogitsLoss
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.0005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()

        # Train metrics
        self.train_loss_acc = LossAccumulator()

        # Validation metrics
        self.val_loss_acc = LossAccumulator()
        self.val_pr_auc = BinaryAveragePrecision()
        self.val_f1 = BinaryF1Score()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

    def setup(self, stage):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "item_idx_to_category_idx"):
            src = dm.item_idx_to_category_idx
            n = min(len(src), len(self.net.item_idx_to_category_idx))
            self.net.item_idx_to_category_idx[:n] = src[:n]

    def forward(self, features, source_ids, target_ids):
        return self.net(features, source_ids, target_ids)

    def training_step(self, batch, batch_idx):
        features, source_ids, target_ids, labels = batch
        logits = self.net(features, source_ids, target_ids)
        loss = self.criterion(logits.squeeze(-1), labels)
        self.train_loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.train_loss_acc.compute()
        self.log("train/loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        features, source_ids, target_ids, labels = batch
        logits = self.net(features, source_ids, target_ids)
        loss = self.criterion(logits.squeeze(-1), labels)

        preds = logits.squeeze(-1)
        self.val_loss_acc.update(loss)
        self.val_pr_auc.update(preds, labels.long())
        self.val_f1.update(preds, labels.long())
        self.val_accuracy.update(preds, labels.long())
        self.val_precision.update(preds, labels.long())
        self.val_recall.update(preds, labels.long())

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", self.val_loss_acc.compute(), prog_bar=True)
        self.log("val/pr_auc", self.val_pr_auc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/accuracy", self.val_accuracy.compute())
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())

    def test_step(self, batch, batch_idx):
        features, source_ids, target_ids, labels = batch
        logits = self.net(features, source_ids, target_ids)

        preds = logits.squeeze(-1)
        self.val_pr_auc.update(preds, labels.long())
        self.val_f1.update(preds, labels.long())
        self.val_accuracy.update(preds, labels.long())
        self.val_precision.update(preds, labels.long())
        self.val_recall.update(preds, labels.long())

    def on_test_epoch_end(self) -> None:
        self.log("test/pr_auc", self.val_pr_auc.compute())
        self.log("test/f1", self.val_f1.compute())
        self.log("test/accuracy", self.val_accuracy.compute())
        self.log("test/precision", self.val_precision.compute())
        self.log("test/recall", self.val_recall.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
