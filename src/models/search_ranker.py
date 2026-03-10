import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from torchmetrics.classification import BinaryAUROC
from torchmetrics import Metric

# Label constants
NEUTRAL = -1  # ignore in loss computation

TASK_NAMES = ["retriever", "ctr", "ctcar", "ctcvr"]
BCE_TASK_NAMES = ["ctr", "ctcar", "ctcvr"]


class ClickNDCG(Metric):
    """NDCG metric for retriever: measures ranking quality of clicked items."""

    full_state_update = False

    def __init__(self, top_k: int = 10):
        super().__init__()
        self.top_k = top_k
        self.add_state("ndcg_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, scores: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            scores: (batch, num_candidates) predicted relevance scores
            targets: (batch, num_candidates) multi-hot or graded relevance
        """
        batch_size = scores.size(0)
        _, topk_indices = scores.topk(self.top_k, dim=-1)  # (batch, k)

        # Gather relevance at predicted top-k positions
        predicted_rel = targets.gather(1, topk_indices)  # (batch, k)

        # DCG
        positions = torch.arange(1, self.top_k + 1, device=scores.device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1)  # (k,)
        dcg = (predicted_rel * discounts.unsqueeze(0)).sum(dim=-1)  # (batch,)

        # Ideal DCG: sort true relevance descending, take top-k
        ideal_rel, _ = targets.sort(dim=-1, descending=True)
        ideal_rel = ideal_rel[:, : self.top_k]
        # Pad if fewer than top_k items
        if ideal_rel.size(1) < self.top_k:
            pad = torch.zeros(
                batch_size, self.top_k - ideal_rel.size(1),
                device=scores.device, dtype=ideal_rel.dtype,
            )
            ideal_rel = torch.cat([ideal_rel, pad], dim=1)
        idcg = (ideal_rel * discounts.unsqueeze(0)).sum(dim=-1)  # (batch,)

        # Avoid division by zero
        valid = idcg > 0
        ndcg = torch.where(valid, dcg / idcg, torch.zeros_like(dcg))

        self.ndcg_sum += ndcg.sum()
        self.total += valid.sum()

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        return self.ndcg_sum / self.total


class SearchRankerModel(L.LightningModule):
    """Multi-task search ranker LightningModule.

    Wraps SearchRankerNet and handles:
    - Multi-task loss: BCE for ctr/ctcar/ctcvr, softmax CE for retriever
    - AUROC metrics per BCE task + NDCG for retriever
    - Derived probabilities: CTCAR = p_ctr * p_c2a, CTCVR = p_ctcar * p_a2o
    - Adam optimizer with LambdaLR scheduling
    """

    def __init__(
        self,
        net: nn.Module,
        task_weight_ctr: float = 1.0,
        task_weight_ctcar: float = 1.0,
        task_weight_ctcvr: float = 1.0,
        softmax_loss_weight_click: float = 1.0,
        softmax_temperature: float = 1.0,
        lr: float = 0.0001,
        best_metric: str = "test_loss",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        self.task_weights = {
            "ctr": task_weight_ctr,
            "ctcar": task_weight_ctcar,
            "ctcvr": task_weight_ctcvr,
        }
        self.softmax_loss_weight_click = softmax_loss_weight_click
        self.softmax_temperature = softmax_temperature
        self.lr = lr

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Metrics: AUROC per BCE task, NDCG for retriever
        for stage in ["val", "test"]:
            for task in BCE_TASK_NAMES:
                setattr(self, f"{stage}_auroc_{task}", BinaryAUROC())
            setattr(self, f"{stage}_click_ndcg", ClickNDCG(top_k=10))

    def forward(self, click_items, click_markets, click_categories, query_tokens, candidate_items):
        return self.net(click_items, click_markets, click_categories, query_tokens, candidate_items)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        task_scores = self.net(
            inputs["click_items"],
            inputs["click_markets"],
            inputs["click_categories"],
            inputs["query_tokens"],
            inputs["candidate_items"],
        )

        loss = self._compute_loss(task_scores, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        task_scores = self.net(
            inputs["click_items"],
            inputs["click_markets"],
            inputs["click_categories"],
            inputs["query_tokens"],
            inputs["candidate_items"],
        )

        loss = self._compute_loss(task_scores, labels)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        self._update_metrics("val", task_scores, labels)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        task_scores = self.net(
            inputs["click_items"],
            inputs["click_markets"],
            inputs["click_categories"],
            inputs["query_tokens"],
            inputs["candidate_items"],
        )

        loss = self._compute_loss(task_scores, labels)
        self.log("test/loss", loss, sync_dist=True)

        self._update_metrics("test", task_scores, labels)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_lambda(epoch):
            # Constant LR
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # --- Internal helpers ---

    def _compute_loss(self, task_scores, labels):
        """Compute weighted multi-task loss.

        - retriever: softmax cross-entropy with multi-hot targets / temperature
        - ctr, ctcar, ctcvr: BCE with neutral label filtering

        Derived labels:
            CTCAR prob = p_ctr * p_click_to_action (but loss is direct BCE on ctcar label)
            CTCVR prob = p_ctcar * p_action_to_order (but loss is direct BCE on ctcvr label)
        """
        total_loss = torch.tensor(0.0, device=task_scores["ctr"].device)

        # --- Retriever loss: softmax CE with multi-hot targets ---
        retriever_scores = task_scores["retriever"]  # (batch, num_candidates)
        retriever_targets = labels["retriever"]       # (batch, num_candidates) multi-hot
        # Only compute for samples that have at least one positive target
        has_positive = retriever_targets.sum(dim=-1) > 0  # (batch,)
        if has_positive.any():
            scores_pos = retriever_scores[has_positive] / self.softmax_temperature
            targets_pos = retriever_targets[has_positive].float()
            # Normalize targets to probability distribution
            target_probs = targets_pos / targets_pos.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # Softmax cross-entropy: -sum(target * log_softmax(scores))
            log_probs = F.log_softmax(scores_pos, dim=-1)
            retriever_loss = -(target_probs * log_probs).sum(dim=-1).mean()
            total_loss = total_loss + self.softmax_loss_weight_click * retriever_loss

        # --- BCE tasks: ctr, ctcar, ctcvr ---
        for task_name in BCE_TASK_NAMES:
            task_logits = task_scores[task_name]  # (batch, num_candidates)
            task_labels = labels[task_name]        # (batch, num_candidates)
            mask = task_labels != NEUTRAL
            if mask.sum() == 0:
                continue

            task_loss = self.bce(task_logits[mask], task_labels[mask].float())
            task_loss = task_loss.mean()
            total_loss = total_loss + self.task_weights[task_name] * task_loss

        return total_loss

    def _update_metrics(self, stage: str, task_scores: dict, labels: dict):
        """Update AUROC for BCE tasks and NDCG for retriever."""
        # BCE tasks: per-element AUROC (flatten across candidates)
        for task in BCE_TASK_NAMES:
            label = labels[task]
            scores = task_scores[task]
            mask = label != NEUTRAL
            if mask.sum() == 0:
                continue
            metric = getattr(self, f"{stage}_auroc_{task}")
            preds = torch.sigmoid(scores[mask])
            metric.update(preds, label[mask].long())

        # Retriever: NDCG
        ndcg_metric = getattr(self, f"{stage}_click_ndcg")
        ndcg_metric.update(task_scores["retriever"], labels["retriever"].float())

    def _log_metrics(self, stage: str):
        """Compute and log all metrics."""
        for task in BCE_TASK_NAMES:
            metric = getattr(self, f"{stage}_auroc_{task}")
            try:
                value = metric.compute()
                self.log(f"{stage}/auroc_{task}", value, prog_bar=(task == "ctr"))
            except ValueError:
                pass
            metric.reset()

        ndcg_metric = getattr(self, f"{stage}_click_ndcg")
        try:
            value = ndcg_metric.compute()
            self.log(f"{stage}/click_ndcg", value, prog_bar=True)
        except ValueError:
            pass
        ndcg_metric.reset()
