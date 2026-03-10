import numpy as np
import torch
import torch.nn as nn


class MultiInterestNet(nn.Module):
    """Multi-Interest Attention Network for sequential recommendation.

    Identifies K distinct user interests via attention over click history,
    then scores all items against the selected (training) or all (inference)
    interest embeddings.

    Input features:
        - click_items:          (B, T)  item indices (1-based, 0=pad)
        - event_codes:          (B, T)  event codes  (0=pad, 1=click, 2=preference)
        - standard_categories:  (B, T)  category indices (1-based, 0=pad)
        - positive_items:       (B,)    target item index (training only)

    Output:
        Training : (B, num_items) logits for the selected interest
        Inference: (B, K, num_items) logits for all interests
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_size: int = 128,
        attention_size: int = 8,
        interest_count: int = 4,
        max_len: int = 80,
        use_item_bias: bool = True,
        soft_selection: bool = False,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.interest_count = interest_count
        self.max_len = max_len
        self.use_item_bias = use_item_bias
        self.soft_selection = soft_selection

        std = 0.1 / np.sqrt(embedding_size)

        # Embeddings (padding_idx=0 for all)
        self.item_embeddings = nn.Embedding(num_items, embedding_size, padding_idx=0)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std=std)

        self.standard_category_embeddings = nn.Embedding(num_categories, embedding_size, padding_idx=0)
        nn.init.normal_(self.standard_category_embeddings.weight, mean=0, std=std)

        self.event_embeddings = nn.Embedding(3, embedding_size, padding_idx=0)
        nn.init.normal_(self.event_embeddings.weight, mean=0, std=std)

        self.position_embeddings = nn.Embedding(max_len, embedding_size)
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=std)

        if use_item_bias:
            self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
            nn.init.normal_(self.item_bias.weight, mean=0, std=0.1)

        # Attention layers
        self.W1 = nn.Linear(embedding_size, attention_size)
        self.W2 = nn.Linear(attention_size, interest_count)

    def forward(
        self,
        click_items: torch.Tensor,
        event_codes: torch.Tensor,
        standard_categories: torch.Tensor,
        positive_items: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            click_items:          (B, T)
            event_codes:          (B, T)
            standard_categories:  (B, T)
            positive_items:       (B,)  required during training

        Returns:
            Training : (B, num_items) scores for selected interest
            Inference: (B, K, num_items) scores for all interests
        """
        valid_history = (click_items > 0).long()  # (B, T)
        batch_size = click_items.size(0)

        # Embedding lookup
        history_item_emb = self.item_embeddings(click_items)              # (B, T, D)
        history_event_emb = self.event_embeddings(event_codes)            # (B, T, D)
        history_cat_emb = self.standard_category_embeddings(standard_categories)  # (B, T, D)

        # Augmented history embedding = item + event + category + position
        history_emb_aug = (
            history_item_emb
            + history_event_emb
            + history_cat_emb
            + self.position_embeddings.weight[None, :, :]  # broadcast (1, T, D)
        )  # (B, T, D)

        # Multi-head attention  ->  (B, T, K)
        attention_score = self.W2(self.W1(history_emb_aug).tanh())
        attention_score = attention_score.masked_fill(valid_history.unsqueeze(-1) == 0, -np.inf)
        attention_score = attention_score.transpose(-1, -2)               # (B, K, T)
        attention_score = (attention_score - attention_score.max()).softmax(dim=-1)
        attention_score = attention_score.masked_fill(torch.isnan(attention_score), 0)

        # Weighted sum of (item + category) embeddings  ->  interest embeddings
        interest_emb = (
            (history_item_emb + history_cat_emb)[:, None, :, :]           # (B, 1, T, D)
            * attention_score[:, :, :, None]                               # (B, K, T, 1)
        ).sum(-2)  # (B, K, D)

        # All item embeddings for scoring
        item_emb = self.item_embeddings.weight                            # (V, D)

        if self.training:
            assert positive_items is not None, "positive_items required during training"
            # Select interest closest to target item
            target_emb = self.item_embeddings(positive_items).squeeze()   # (B, D)
            target_pred = (interest_emb * target_emb[:, None, :]).sum(-1) # (B, K)
            if self.soft_selection:
                target_prob = torch.softmax(target_pred, dim=-1)
                idx_select = torch.multinomial(target_prob, 1).flatten()
            else:
                idx_select = target_pred.max(-1)[1]                       # (B,)
            user_emb = interest_emb[torch.arange(batch_size), idx_select, :]  # (B, D)
            prediction = torch.matmul(user_emb, item_emb.T)              # (B, V)
        else:
            # All interests
            prediction = torch.matmul(interest_emb, item_emb.T)          # (B, K, V)

        if self.use_item_bias:
            prediction = prediction + self.item_bias.weight.squeeze(-1)   # broadcast bias

        return prediction

    @torch.no_grad()
    def compute_score(
        self,
        click_items: torch.Tensor,
        event_codes: torch.Tensor,
        standard_categories: torch.Tensor,
        top_k: int = 50,
        remove_history: bool = True,
    ) -> torch.Tensor:
        """Inference scoring with interest aggregation.

        For each interest, rank items, then aggregate across interests using
        score = interest_norm / (1 + rank), summed over interests.

        Args:
            click_items:          (B, T)
            event_codes:          (B, T)
            standard_categories:  (B, T)
            top_k:                number of items to rank per interest
            remove_history:       zero-out history items before ranking

        Returns:
            final_scores: (B, num_items) aggregated scores
        """
        was_training = self.training
        self.eval()

        prediction = self.forward(click_items, event_codes, standard_categories)  # (B, K, V)
        prediction = prediction.softmax(dim=-1)

        batch_size, interest_count, item_count = prediction.shape

        # Interest magnitude as weight
        # We don't have interest_emb here directly, so recompute it
        valid_history = (click_items > 0).long()
        history_item_emb = self.item_embeddings(click_items)
        history_event_emb = self.event_embeddings(event_codes)
        history_cat_emb = self.standard_category_embeddings(standard_categories)

        history_emb_aug = (
            history_item_emb
            + history_event_emb
            + history_cat_emb
            + self.position_embeddings.weight[None, :, :]
        )

        attention_score = self.W2(self.W1(history_emb_aug).tanh())
        attention_score = attention_score.masked_fill(valid_history.unsqueeze(-1) == 0, -np.inf)
        attention_score = attention_score.transpose(-1, -2)
        attention_score = (attention_score - attention_score.max()).softmax(dim=-1)
        attention_score = attention_score.masked_fill(torch.isnan(attention_score), 0)

        interest_emb = (
            (history_item_emb + history_cat_emb)[:, None, :, :] * attention_score[:, :, :, None]
        ).sum(-2)

        interest_score = torch.linalg.vector_norm(interest_emb, dim=2)  # (B, K)

        if remove_history:
            for i in range(batch_size):
                for j in range(interest_count):
                    prediction[i][j][click_items[i]] = -np.inf

        top_k = min(top_k, item_count)
        _, ranked_items = prediction.topk(k=top_k)  # (B, K, top_k)
        default_score_at_rank = 1.0 + torch.arange(top_k, device=prediction.device, dtype=torch.float32)

        scores = torch.zeros_like(prediction)
        for i in range(batch_size):
            for j in range(interest_count):
                scores[i][j][ranked_items[i][j]] = interest_score[i][j] / default_score_at_rank

        final_scores = scores.sum(dim=1)  # (B, V)

        if was_training:
            self.train()

        return final_scores
