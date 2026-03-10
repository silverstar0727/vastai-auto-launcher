import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BanditNet(nn.Module):
    """Neural Linear Bandit using frozen two-tower user embeddings.

    For each item, maintains a Bayesian linear regression model:
        B_mat:      (num_items, D, D) precision matrices, initialized as lambda * I
        item_Y_sum: (num_items, D)    reward-weighted feature sum

    Training (single epoch):
        - Accumulate outer products of user embeddings weighted by labels
        - B_mat[item] += user_emb @ user_emb.T  (for each observed item)
        - Y_sum[item] += label * user_emb

    Post-training:
        - theta[item] = inv(B_mat[item]) @ Y_sum[item]    (MAP estimate)
        - cov_L[item] = cholesky(inv(B_mat[item]))         (for Thompson sampling)

    Inference (Thompson Sampling):
        - sample_theta = theta + explore_weight * cov_L @ z,  z ~ N(0, I)
        - score = user_emb @ sample_theta.T
    """

    def __init__(
        self,
        frozen_two_tower: nn.Module,
        num_items: int,
        embed_dim: int = 32,
        lambda_value: float = 200.0,
    ):
        super().__init__()
        self.frozen_two_tower = frozen_two_tower
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.lambda_value = lambda_value

        # Freeze the two-tower model
        for param in self.frozen_two_tower.parameters():
            param.requires_grad = False
        self.frozen_two_tower.eval()

        # Precision matrices: B_mat[i] = lambda * I initially
        B_init = lambda_value * torch.eye(embed_dim).unsqueeze(0).expand(num_items, -1, -1).clone()
        self.register_buffer("B_mat", B_init)

        # Reward-weighted feature sum
        self.register_buffer("item_Y_sum", torch.zeros(num_items, embed_dim))

        # Computed parameters (populated after training via compute_parameters)
        self.register_buffer("theta", torch.zeros(num_items, embed_dim))
        self.register_buffer("cov_L", torch.zeros(num_items, embed_dim, embed_dim))

    @torch.no_grad()
    def accumulate(self, user_emb: torch.Tensor, item_ids: torch.Tensor, labels: torch.Tensor):
        """Accumulate sufficient statistics from a batch.

        Args:
            user_emb: (N, D) user embeddings from frozen two-tower
            item_ids: (N,) item indices (0-indexed into num_items)
            labels:   (N,) binary labels (1=click, 0=no-click)
        """
        for i in range(user_emb.size(0)):
            item_id = item_ids[i].item()
            if item_id < 0 or item_id >= self.num_items:
                continue
            u = user_emb[i]  # (D,)
            # B_mat[item] += u @ u.T
            self.B_mat[item_id] += torch.outer(u, u)
            # Y_sum[item] += label * u
            self.item_Y_sum[item_id] += labels[i].float() * u

    @torch.no_grad()
    def compute_parameters(self, chunk_size: int = 1000):
        """Compute theta and cov_L from accumulated statistics.

        Processes items in chunks to avoid OOM on large item sets.
            theta[i] = inv(B[i]) @ Y_sum[i]
            cov_L[i] = cholesky(inv(B[i]))
        """
        device = self.B_mat.device
        num_computed = 0
        num_failed = 0

        for start in range(0, self.num_items, chunk_size):
            end = min(start + chunk_size, self.num_items)
            B_chunk = self.B_mat[start:end]  # (C, D, D)
            Y_chunk = self.item_Y_sum[start:end]  # (C, D)

            try:
                B_inv = torch.linalg.inv(B_chunk)  # (C, D, D)
            except torch.linalg.LinAlgError:
                # Fallback: process one by one
                for j in range(end - start):
                    idx = start + j
                    try:
                        B_inv_j = torch.linalg.inv(self.B_mat[idx])
                        self.theta[idx] = B_inv_j @ self.item_Y_sum[idx]
                        self.cov_L[idx] = torch.linalg.cholesky(B_inv_j)
                        num_computed += 1
                    except torch.linalg.LinAlgError:
                        # Item has no observations; keep defaults (zeros)
                        num_failed += 1
                continue

            # theta = inv(B) @ Y_sum
            self.theta[start:end] = torch.bmm(
                B_inv, Y_chunk.unsqueeze(-1)
            ).squeeze(-1)  # (C, D)

            # cov_L = cholesky(inv(B))
            try:
                self.cov_L[start:end] = torch.linalg.cholesky(B_inv)
            except torch.linalg.LinAlgError:
                # Fallback: one by one for cholesky failures
                for j in range(end - start):
                    idx = start + j
                    try:
                        self.cov_L[idx] = torch.linalg.cholesky(
                            torch.linalg.inv(self.B_mat[idx])
                        )
                    except torch.linalg.LinAlgError:
                        # Add small jitter for numerical stability
                        jitter = 1e-6 * torch.eye(self.embed_dim, device=device)
                        try:
                            self.cov_L[idx] = torch.linalg.cholesky(
                                torch.linalg.inv(self.B_mat[idx]) + jitter
                            )
                        except torch.linalg.LinAlgError:
                            num_failed += 1
                            continue
                    num_computed += 1
                continue

            num_computed += (end - start)

        logger.info(
            f"Bandit parameters computed: {num_computed} items succeeded, "
            f"{num_failed} items failed (kept at zero)"
        )

    @torch.no_grad()
    def thompson_sample(
        self,
        user_emb: torch.Tensor,
        candidate_items: torch.Tensor,
        explore_weight: float = 1.0,
    ) -> torch.Tensor:
        """Thompson Sampling scores for candidate items.

        Args:
            user_emb:        (N, D) user embeddings
            candidate_items: (K,) candidate item indices
            explore_weight:  scaling factor for exploration noise

        Returns:
            scores: (N, K) Thompson-sampled scores
        """
        K = candidate_items.size(0)
        D = self.embed_dim

        # Gather parameters for candidates
        theta_cand = self.theta[candidate_items]  # (K, D)
        cov_L_cand = self.cov_L[candidate_items]  # (K, D, D)

        # Sample noise: z ~ N(0, I)
        z = torch.randn(K, D, device=user_emb.device)  # (K, D)

        # Perturbed theta: theta + explore_weight * cov_L @ z
        noise = torch.bmm(cov_L_cand, z.unsqueeze(-1)).squeeze(-1)  # (K, D)
        sampled_theta = theta_cand + explore_weight * noise  # (K, D)

        # Score: user_emb @ sampled_theta.T
        scores = torch.matmul(user_emb, sampled_theta.T)  # (N, K)
        return scores

    def forward(self, user_features, candidate_items, explore_weight=1.0):
        """End-to-end: extract user embeddings from frozen two-tower, then Thompson sample.

        Args:
            user_features: dict for frozen two-tower user tower
            candidate_items: (K,) candidate item indices
            explore_weight: exploration scaling

        Returns:
            scores: (N, K)
        """
        self.frozen_two_tower.eval()
        user_emb = self.frozen_two_tower.forward_user_tower(user_features)
        return self.thompson_sample(user_emb, candidate_items, explore_weight)
