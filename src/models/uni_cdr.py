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

    def update(self, loss: torch.Tensor):
        self.loss += loss.detach()
        self.total += 1

    def compute(self):
        return self.loss / self.total


class AccuracyAndCoverage(Metric):
    """NDCG@K, Recall@K, and Coverage@K for recommendation evaluation."""

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


# ---------------------------------------------------------------------------
# User type constants (must match dataset)
# ---------------------------------------------------------------------------

USER_COMMON = 0
USER_SOURCE = 1
USER_TARGET = 2


class UniCDRModel(L.LightningModule):
    """Lightning module for Unified Cross-Domain Recommendation.

    7 loss components:
        1. src_ce:       CE on source within-domain scores
        2. target_ce:    CE on target within-domain scores
        3. shared_src:   CE on shared-user source scores
        4. shared_target: CE on shared-user target scores
        5. masking_src:  CE on cross-domain source scores
        6. masking_target: CE on cross-domain target scores
        7. contrastive:  BCE on positive/negative user pairs across domains
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.001,
        weight_src_loss: float = 1.0,
        weight_target_loss: float = 1.0,
        weight_contrastive_loss: float = 1.0,
        top_k: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.lr = lr
        self.weight_src_loss = weight_src_loss
        self.weight_target_loss = weight_target_loss
        self.weight_contrastive_loss = weight_contrastive_loss

        # Metrics
        target_num_items = net.target_num_items
        self.train_loss_acc = LossAccumulator()
        self.val_metric = AccuracyAndCoverage(num_items=target_num_items, top_k=top_k)
        self.test_metric = AccuracyAndCoverage(num_items=target_num_items, top_k=top_k)

    def forward(self, src_inputs, target_inputs):
        return self.net(src_inputs, target_inputs)

    def training_step(self, batch, batch_idx):
        src_inputs, target_inputs, user_types, src_labels, target_labels = batch
        outputs = self.net(src_inputs, target_inputs)

        loss = self._compute_loss(outputs, user_types, src_labels, target_labels)
        self.train_loss_acc.update(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        avg_loss = self.train_loss_acc.compute()
        self.log("train/loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        src_inputs, target_inputs, user_types, src_labels, target_labels = batch
        outputs = self.net(src_inputs, target_inputs)

        # Evaluate on target domain scores
        self.val_metric.update(outputs["target_scores"], target_labels)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        src_inputs, target_inputs, user_types, src_labels, target_labels = batch
        outputs = self.net(src_inputs, target_inputs)

        self.test_metric.update(outputs["target_scores"], target_labels)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metric.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        user_types: torch.Tensor,
        src_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        # Masks for user types
        has_src = (user_types == USER_COMMON) | (user_types == USER_SOURCE)
        has_target = (user_types == USER_COMMON) | (user_types == USER_TARGET)
        is_common = user_types == USER_COMMON

        total_loss = torch.tensor(0.0, device=self.device)

        # 1. Source CE loss
        if has_src.any():
            src_ce = F.cross_entropy(
                outputs["src_scores"][has_src], src_labels[has_src]
            )
            total_loss = total_loss + self.weight_src_loss * src_ce

        # 2. Target CE loss
        if has_target.any():
            target_ce = F.cross_entropy(
                outputs["target_scores"][has_target], target_labels[has_target]
            )
            total_loss = total_loss + self.weight_target_loss * target_ce

        # 3-4. Shared user CE losses (only for common users)
        if is_common.any():
            shared_src_ce = F.cross_entropy(
                outputs["src_by_shared"][is_common], src_labels[is_common]
            )
            shared_target_ce = F.cross_entropy(
                outputs["target_by_shared"][is_common], target_labels[is_common]
            )
            total_loss = total_loss + shared_src_ce + shared_target_ce

        # 5-6. Masking (cross-domain) CE losses (only for common users)
        if is_common.any():
            masking_src_ce = F.cross_entropy(
                outputs["src_by_masking"][is_common], src_labels[is_common]
            )
            masking_target_ce = F.cross_entropy(
                outputs["target_by_masking"][is_common], target_labels[is_common]
            )
            total_loss = total_loss + masking_src_ce + masking_target_ce

        # 7. Contrastive loss across domains for common users
        if is_common.sum() >= 2:
            contra_loss = self._contrastive_loss(
                outputs["src_user_emb"][is_common],
                outputs["target_user_emb"][is_common],
            )
            total_loss = total_loss + self.weight_contrastive_loss * contra_loss

        return total_loss

    def _contrastive_loss(
        self,
        src_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """BCE contrastive loss: same-user pairs are positive, others negative.

        Args:
            src_emb: (M, D) source user embeddings for common users
            target_emb: (M, D) target user embeddings for common users
        """
        # Similarity matrix (M, M)
        sim_matrix = torch.mm(src_emb, target_emb.t())

        # Positive pairs: diagonal (same user across domains)
        # Negative pairs: off-diagonal
        M = sim_matrix.size(0)
        labels = torch.eye(M, device=sim_matrix.device)

        loss = F.binary_cross_entropy_with_logits(sim_matrix, labels)
        return loss
