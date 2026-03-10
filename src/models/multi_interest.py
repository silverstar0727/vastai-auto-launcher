import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AccuracyAndCoverage(Metric):
    full_state_update = False

    def __init__(self, num_items: int, top_k: int = 10):
        super().__init__()
        self.add_state("total", default=torch.tensor(0))
        self.add_state("ndcg", default=torch.tensor(0.0))
        self.add_state("recall", default=torch.tensor(0.0))
        self.top_k = top_k
        self.num_items = num_items
        self.cover_items = set()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_rank_items = preds.argsort(dim=1, descending=True)[:, : self.top_k]
        metrics = self._recalls_and_ndcgs(batch_rank_items, target)
        self.ndcg += metrics["ndcg"]
        self.recall += metrics["recall"]
        self.total += 1
        self.cover_items.update(batch_rank_items.cpu().numpy().reshape(-1).tolist())

    def compute(self):
        scores = {
            f"NDCG_{self.top_k}": self.ndcg / self.total,
            f"Recall_{self.top_k}": self.recall / self.total,
            f"Coverage_{self.top_k}": len(self.cover_items) / self.num_items,
        }
        self.cover_items = set()
        return scores

    def _recalls_and_ndcgs(self, batch_rank_items, batch_positive_items):
        batch_size = batch_rank_items.size(0)
        recall_sum = 0.0
        ndcg_sum = 0.0

        for i in range(batch_size):
            rank_items = batch_rank_items[i]
            positive_items = batch_positive_items[i]
            positive_items = positive_items[positive_items > 0]

            hit_mask = torch.isin(rank_items, positive_items)
            num_hits = hit_mask.sum().float()
            num_positives = max(positive_items.numel(), 1)

            recall_sum += (num_hits / num_positives).item()

            if num_hits > 0:
                positions = torch.where(hit_mask)[0].float() + 1
                dcg = (1.0 / torch.log2(positions + 1)).sum()
                ideal_positions = torch.arange(1, num_hits.int().item() + 1, dtype=torch.float32)
                idcg = (1.0 / torch.log2(ideal_positions + 1)).sum()
                ndcg_sum += (dcg / idcg).item()

        return {
            "recall": recall_sum / batch_size,
            "ndcg": ndcg_sum / batch_size,
        }


class MultiInterestModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        top_k: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        lr_scheduler: str = "step",
        lr_scheduler_step: int = 4,
        lr_scheduler_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        num_items = net.num_items
        self.test_metric = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.loss_acc = LossAccumulator()

    def forward(self, click_items, event_codes, standard_categories, positive_items=None):
        return self.net(click_items, event_codes, standard_categories, positive_items)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        scores = self.net(
            inputs["click_items"],
            inputs["event_codes"],
            inputs["standard_categories"],
            positive_items=labels,
        )
        loss = _loss_fn(scores, labels)
        self.loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_scores = self.net.compute_score(
            inputs["click_items"],
            inputs["event_codes"],
            inputs["standard_categories"],
            top_k=50,
            remove_history=True,
        )
        self.test_metric.update(batch_scores, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_scores = self.net.compute_score(
            inputs["click_items"],
            inputs["event_codes"],
            inputs["standard_categories"],
            top_k=50,
            remove_history=True,
        )
        self.test_metric.update(batch_scores, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if not self.hparams.lr_scheduler:
            return optimizer

        if self.hparams.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.lr_scheduler_step,
                gamma=self.hparams.lr_scheduler_gamma,
            )
        elif self.hparams.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.lr_scheduler_gamma,
            )
        else:
            return optimizer

        return [optimizer], [scheduler]


def _loss_fn(logits, labels):
    """Cross-entropy loss with ignore_index=0 (padding)."""
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels, ignore_index=0)
