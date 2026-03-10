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
        loss = self.loss / self.total
        self.reset()
        return loss


class AccuracyAndCoverage(Metric):
    full_state_update = False

    def __init__(self, num_items: int, top_k: int = 10):
        super().__init__()
        self.add_state("total", default=torch.tensor(0))
        self.add_state("ndcg", default=torch.tensor(0.0))
        self.add_state("recall", default=torch.tensor(0.0))
        self.add_state("cover_mask", default=torch.zeros(num_items + 1, dtype=torch.bool))
        self.top_k = top_k
        self.num_items = num_items

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_rank_items = preds.argsort(dim=1, descending=True)[:, : self.top_k]
        metrics = self._recalls_and_ndcgs(batch_rank_items, target)
        self.ndcg += metrics["ndcg"]
        self.recall += metrics["recall"]
        self.total += 1
        self.cover_mask[batch_rank_items.flatten()] = True

    def compute(self):
        scores = {
            f"NDCG_{self.top_k}": self.ndcg / self.total,
            f"Recall_{self.top_k}": self.recall / self.total,
            f"Coverage_Percentage_{self.top_k}": self.cover_mask.sum().float() / self.num_items,
            f"Coverage_Counts_{self.top_k}": self.cover_mask.sum().float(),
        }
        self.reset()
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


class BertModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        top_k: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        num_items = net.num_items
        self.test_metric = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.test_metric_one_click = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.loss_acc = LossAccumulator()

    def setup(self, stage):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return
        if hasattr(dm, "num_items") and dm.num_items > 0 and dm.num_items != self.net.num_items:
            raise ValueError(
                f"num_items mismatch: model config has {self.net.num_items}, "
                f"but data produced {dm.num_items}. "
                f"Update num_items in bert.yaml to match."
            )
        if hasattr(dm, "text_embeddings"):
            self.net.set_text_embeddings(dm.text_embeddings)

    def forward(self, item_indexes, events, time_intervals):
        return self.net(item_indexes, events, time_intervals)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        scores = self.net(inputs["click_items"], inputs["event_code"], inputs["time_interval"])
        loss = _loss_fn(scores, labels)
        self.loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_scores = self.net(inputs["click_items"], inputs["event_code"], inputs["time_interval"])[:, -1, :]
        self.test_metric.update(batch_scores, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_scores = self.net(inputs["click_items"], inputs["event_code"], inputs["time_interval"])[:, -1, :]
        self.test_metric.update(batch_scores, targets)
        self._one_click_test(inputs, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        one_click_metrics = self.test_metric_one_click.compute()
        for k, v in one_click_metrics.items():
            self.log(f"test/one_click_{k}", v)

    def _one_click_test(self, inputs, targets):
        items = inputs["click_items"].clone()
        events = inputs["event_code"].clone()
        times = inputs["time_interval"].clone()
        items[:, :-2] = 0
        events[:, :-2] = 0
        times[:, :-2] = 0
        scores = self.net(items, events, times)[:, -1, :]
        self.test_metric_one_click.update(scores, targets)


def _loss_fn(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels, ignore_index=0)
