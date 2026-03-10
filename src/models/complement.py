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


class ComplementModel(L.LightningModule):
    """Complement recommendation LightningModule.

    Wraps ComplementNet and handles:
        - Training with softmax cross-entropy (negative_sampling=False by default)
        - Validation / test with NDCG@50, Recall@50, Coverage@50
        - Optimizer configuration (Adam)
    """

    def __init__(
        self,
        net: nn.Module,
        num_items: int,
        negative_sampling: bool = False,
        top_k: int = 50,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        self.negative_sampling = negative_sampling

        self.val_metric = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.test_metric = AccuracyAndCoverage(num_items=num_items, top_k=top_k)
        self.loss_acc = LossAccumulator()

    def setup(self, stage):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            if hasattr(dm, "text_embeddings"):
                self.net.set_text_embeddings(dm.text_embeddings)
            if hasattr(dm, "item_markets") and hasattr(dm, "item_categories"):
                self.net.set_item_features(
                    item_markets=dm.item_markets,
                    item_categories=dm.item_categories,
                )

    def forward(self, user_features, item_idxes=None):
        return self.net(user_features, item_idxes)

    def training_step(self, batch, batch_idx):
        if self.negative_sampling:
            inputs, batch_pos_labels = batch
            in_batch_pos_items = torch.unique(batch_pos_labels.reshape(-1))

            pred_scores = self.net(inputs, in_batch_pos_items)
            target_scores = _create_target_scores(pred_scores, batch_pos_labels, in_batch_pos_items)
            loss = -torch.mean(torch.sum(F.log_softmax(pred_scores, dim=1) * target_scores, dim=-1))
        else:
            inputs, labels = batch
            scores = self.net(inputs)

            # Pad labels if net's vocab is larger than data's vocab
            if labels.shape[1] < scores.shape[1]:
                labels = F.pad(labels, (0, scores.shape[1] - labels.shape[1]))

            logits_ignore_unknown = scores[:, 1:]
            targets_ignore_unknown = labels[:, 1:]
            loss = -torch.mean(
                torch.sum(F.log_softmax(logits_ignore_unknown, dim=1) * targets_ignore_unknown, dim=-1)
            )

        self.loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        inputs, batch_pos_labels = batch

        if self.negative_sampling:
            in_batch_items = torch.unique(batch_pos_labels.reshape(-1))
            pred_scores = self.net(inputs, in_batch_items)

            # Remap positive labels to in-batch item indices for metric computation
            targets = []
            for pos_labels in batch_pos_labels:
                pos_items = pos_labels[pos_labels > 0]
                _, pos_loc = (pos_items.reshape(-1, 1) == in_batch_items.reshape(1, -1)).nonzero(as_tuple=True)
                targets.append(pos_loc.reshape(1, -1))
            targets = torch.cat(targets, dim=0)
        else:
            pred_scores = self.net(inputs)
            targets = batch_pos_labels

        self.val_metric.update(pred_scores, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, batch_pos_labels = batch

        if self.negative_sampling:
            in_batch_items = torch.unique(batch_pos_labels.reshape(-1))
            pred_scores = self.net(inputs, in_batch_items)

            targets = []
            for pos_labels in batch_pos_labels:
                pos_items = pos_labels[pos_labels > 0]
                _, pos_loc = (pos_items.reshape(-1, 1) == in_batch_items.reshape(1, -1)).nonzero(as_tuple=True)
                targets.append(pos_loc.reshape(1, -1))
            targets = torch.cat(targets, dim=0)
        else:
            pred_scores = self.net(inputs)
            targets = batch_pos_labels

        self.test_metric.update(pred_scores, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def _create_target_scores(pred_scores, batch_pos_labels, in_batch_pos_items):
    """Build one-hot target tensor aligned to in-batch positive items.

    Args:
        pred_scores: (N, len(in_batch_pos_items))
        batch_pos_labels: (N, max_pos_items) with zero-padding
        in_batch_pos_items: (K,) unique positive items in batch

    Returns:
        target_scores: (N, K) with 1.0 at positive positions
    """
    target_scores = torch.zeros_like(pred_scores)
    for i, pos_labels in enumerate(batch_pos_labels):
        pos_items = pos_labels[pos_labels > 0]
        _, pos_in_target = (pos_items.reshape(-1, 1) == in_batch_pos_items.reshape(1, -1)).nonzero(as_tuple=True)
        target_scores[i, pos_in_target] = 1.0
    return target_scores
