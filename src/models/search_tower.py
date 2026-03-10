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

    def __init__(self, num_items: int, top_k: int = 50):
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


class SearchTowerModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        top_k: int = 50,
        lr: float = 0.001,
        lr_scheduler_step: int = 4,
        lr_scheduler_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.lr = lr
        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_gamma = lr_scheduler_gamma

        num_items = net.num_items
        self.test_metric = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.loss_acc = LossAccumulator()

    def setup(self, stage):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "text_embeddings"):
            self.net.set_text_embeddings(dm.text_embeddings)

    def forward(self, click_items, query_tokens, query_time):
        return self.net(click_items, query_tokens, query_time)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.net(inputs["click_items"], inputs["query_tokens"], inputs["query_time"])
        loss = _softmax_ce_loss(logits, targets)
        self.loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.net(inputs["click_items"], inputs["query_tokens"], inputs["query_time"])
        # For eval, targets is a single positive item index per sample
        self.test_metric.update(logits[:, 1:], targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.net(inputs["click_items"], inputs["query_tokens"], inputs["query_time"])
        self.test_metric.update(logits[:, 1:], targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_scheduler_step,
            gamma=self.lr_scheduler_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _softmax_ce_loss(logits, targets):
    """Softmax cross-entropy with multi-hot targets.

    Args:
        logits: (B, num_items+1) raw scores
        targets: (B, num_items+1) multi-hot target labels (index 0 is padding)
    Returns:
        scalar loss
    """
    log_probs = F.log_softmax(logits[:, 1:], dim=-1)
    target_probs = targets[:, 1:].float()
    loss = -torch.mean(torch.sum(log_probs * target_probs, dim=-1))
    return loss
