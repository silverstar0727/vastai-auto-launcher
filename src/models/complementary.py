import torch
import torch.nn as nn

import lightning as L
from torchmetrics import Metric


class LossAccumulator(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0))
        self.add_state("loss", default=torch.tensor(0.0))

    def update(self, loss):
        self.loss += loss
        self.total += 1

    def compute(self):
        return self.loss / self.total


class ClassificationMetrics(Metric):
    """Accumulates predictions and targets, then computes F1, Accuracy, Precision, Recall (macro)."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)

        # Macro-averaged binary metrics
        metrics = {}
        for cls in [0, 1]:
            tp = ((preds == cls) & (targets == cls)).sum().float()
            fp = ((preds == cls) & (targets != cls)).sum().float()
            fn = ((preds != cls) & (targets == cls)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}

        macro_precision = (metrics[0]["precision"] + metrics[1]["precision"]) / 2
        macro_recall = (metrics[0]["recall"] + metrics[1]["recall"]) / 2
        macro_f1 = (metrics[0]["f1"] + metrics[1]["f1"]) / 2
        accuracy = (preds == targets).float().mean()

        return {
            "f1": macro_f1,
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
        }


class ComplementaryModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.005,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_acc = LossAccumulator()
        self.val_metric = ClassificationMetrics()
        self.test_metric = ClassificationMetrics()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def training_step(self, batch, batch_idx):
        features, labels, weights = batch
        logits = self.net(features).squeeze(-1)  # (N,)
        loss_per_sample = self.criterion(logits, labels.float())
        loss = (loss_per_sample * weights).mean()
        self.loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        features, labels, weights = batch
        logits = self.net(features).squeeze(-1)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        self.val_metric.update(preds, labels)

        # Log val_loss for checkpointing
        loss_per_sample = self.criterion(logits, labels.float())
        loss = (loss_per_sample * weights).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        features, labels, weights = batch
        logits = self.net(features).squeeze(-1)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        self.test_metric.update(preds, labels)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
