import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

logger = logging.getLogger(__name__)


class TwoTowerModel(L.LightningModule):
    """Lightning module for pre-training the Two-Tower network.

    Uses softmax cross-entropy loss: the model outputs temperature-scaled
    logits (N, num_items) and we use cross-entropy against target item indices.
    Same training objective as CBF.
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.0001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, user_features, item_idxes=None):
        return self.net(user_features, item_idxes)

    def training_step(self, batch, batch_idx):
        user_features, target_items = batch
        # target_items: (N,) indices of target items
        # Forward through both towers, get temperature-scaled logits
        logits = self.net(user_features, item_idxes=target_items)  # (N, K)

        # Build labels: each sample's target is at its corresponding position
        # When item_idxes=target_items, logits[i, i] is the score for sample i's target
        # Use identity mapping as labels (diagonal is the correct class)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = self.criterion(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_features, target_items = batch
        logits = self.net(user_features, item_idxes=target_items)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = self.criterion(logits, labels)

        # Compute recall@K
        with torch.no_grad():
            _, topk_indices = logits.topk(min(10, logits.size(1)), dim=1)
            hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
            recall_at_10 = hits.mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall@10", recall_at_10, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        user_features, target_items = batch
        logits = self.net(user_features, item_idxes=target_items)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = self.criterion(logits, labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class BanditModel(L.LightningModule):
    """Lightning module for Neural Linear Bandit training.

    This is a non-gradient model: it accumulates sufficient statistics
    (outer products and reward sums) over a single epoch, then computes
    the Bayesian linear regression parameters.

    Training is a single epoch over display log data.
    """

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.net = net
        # Disable automatic optimization since there are no gradients
        self.automatic_optimization = False

    def forward(self, user_features, candidate_items, explore_weight=1.0):
        return self.net(user_features, candidate_items, explore_weight)

    def training_step(self, batch, batch_idx):
        """Accumulate statistics from a batch of display logs.

        batch: (click_items_dict, item_targets, labels)
            click_items_dict: dict with click_items (N,T) etc. for user tower
            item_targets:     (N,) displayed item indices
            labels:           (N,) binary labels (1=click, 0=no-click)
        """
        click_items_dict, item_targets, labels = batch

        # Get user embeddings from frozen two-tower (no grad)
        with torch.no_grad():
            self.net.frozen_two_tower.eval()
            user_emb = self.net.frozen_two_tower.forward_user_tower(click_items_dict)

        # Accumulate into B_mat and Y_sum
        self.net.accumulate(user_emb, item_targets, labels)

        # Log batch stats
        click_rate = labels.float().mean()
        self.log("train/click_rate", click_rate, on_step=True, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        """Compute theta and cov_L from accumulated statistics."""
        logger.info("Computing bandit parameters from accumulated statistics...")
        self.net.compute_parameters()
        logger.info("Bandit parameter computation complete.")

    def validation_step(self, batch, batch_idx):
        """Evaluate using Thompson Sampling scores."""
        click_items_dict, item_targets, labels = batch

        with torch.no_grad():
            self.net.frozen_two_tower.eval()
            user_emb = self.net.frozen_two_tower.forward_user_tower(click_items_dict)

            # Get unique items in batch for scoring
            unique_items = item_targets.unique()
            scores = self.net.thompson_sample(user_emb, unique_items, explore_weight=0.0)

            # Map item_targets to indices in unique_items
            item_to_idx = {item.item(): idx for idx, item in enumerate(unique_items)}
            target_idx = torch.tensor(
                [item_to_idx[t.item()] for t in item_targets],
                device=scores.device,
            )

            # Compute loss (cross-entropy over unique items)
            loss = F.cross_entropy(scores, target_idx)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test using greedy (no exploration) Thompson Sampling."""
        click_items_dict, item_targets, labels = batch

        with torch.no_grad():
            self.net.frozen_two_tower.eval()
            user_emb = self.net.frozen_two_tower.forward_user_tower(click_items_dict)

            unique_items = item_targets.unique()
            scores = self.net.thompson_sample(user_emb, unique_items, explore_weight=0.0)

            item_to_idx = {item.item(): idx for idx, item in enumerate(unique_items)}
            target_idx = torch.tensor(
                [item_to_idx[t.item()] for t in item_targets],
                device=scores.device,
            )

            loss = F.cross_entropy(scores, target_idx)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        # No optimizer needed (no gradient-based training)
        # Return empty list to satisfy Lightning's requirement
        return []
