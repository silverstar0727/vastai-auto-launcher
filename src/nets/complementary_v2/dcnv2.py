import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    """Single cross layer: x_{l+1} = x_0 * (x_l^T * w_l + b_l) + x_l, then ReLU."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.empty(input_dim))
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.weight.unsqueeze(1))
        nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0, xl: (batch, input_dim)
        interaction = torch.sum(xl * self.weight, dim=-1, keepdim=True)
        out = x0 * (interaction + self.bias.unsqueeze(0)) + xl
        return self.activation(out)


class DeepLayer(nn.Module):
    """Deep part: stack of Linear -> ReLU -> Dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.5):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DCNv2(nn.Module):
    """Deep & Cross Network V2 with Item & Category Embeddings.

    Input:
        - features: (N, 16) z-score normalized behavioral features
        - source_ids: (N,) source item indices
        - target_ids: (N,) target item indices

    The model looks up item embeddings and category embeddings for both
    source and target, computes L2 distance between category embeddings,
    and concatenates everything:
        [features(16) | src_item_emb(512) | tgt_item_emb(512) |
         src_cat_emb(8) | tgt_cat_emb(8) | cat_l2_dist(1)] = 1057

    Output:
        - logits: (N, 1) raw logits for binary classification
    """

    def __init__(
        self,
        input_dim: int = 16,
        num_items: int = 1,
        num_categories: int = 1,
        num_cross_layers: int = 5,
        deep_hidden_dims: list[int] | None = None,
        item_embedding_dim: int = 512,
        category_embedding_dim: int = 8,
        dropout: float = 0.5,
        item_idx_to_category_idx: torch.Tensor | None = None,
    ):
        super().__init__()
        if deep_hidden_dims is None:
            deep_hidden_dims = [512, 128, 64, 16]

        self.input_dim = input_dim
        self.item_embedding_dim = item_embedding_dim
        self.category_embedding_dim = category_embedding_dim

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)

        # Item-to-category mapping (buffer so it moves with the model)
        if item_idx_to_category_idx is None:
            item_idx_to_category_idx = torch.zeros(num_items, dtype=torch.long)
        self.register_buffer("item_idx_to_category_idx", item_idx_to_category_idx)

        # Total concatenated dimension:
        # features + src_item_emb + tgt_item_emb + src_cat_emb + tgt_cat_emb + l2_dist
        total_dim = (
            input_dim
            + item_embedding_dim * 2
            + category_embedding_dim * 2
            + 1
        )

        # Cross network
        self.cross_layers = nn.ModuleList(
            [CrossLayer(total_dim) for _ in range(num_cross_layers)]
        )

        # Deep network
        self.deep_layer = DeepLayer(total_dim, deep_hidden_dims, dropout)

        # Output: concat cross_output + deep_output -> 1
        combined_dim = total_dim + self.deep_layer.output_dim
        self.output = nn.Linear(combined_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Item embeddings: (N, item_embedding_dim)
        src_item_emb = self.item_embedding(source_ids)
        tgt_item_emb = self.item_embedding(target_ids)

        # Category embeddings via item-to-category mapping: (N, category_embedding_dim)
        src_cat_idx = self.item_idx_to_category_idx[source_ids]
        tgt_cat_idx = self.item_idx_to_category_idx[target_ids]
        src_cat_emb = self.category_embedding(src_cat_idx)
        tgt_cat_emb = self.category_embedding(tgt_cat_idx)

        # L2 distance between category embeddings: (N, 1)
        cat_l2_dist = torch.norm(src_cat_emb - tgt_cat_emb, p=2, dim=-1, keepdim=True)

        # Concatenate all: (N, total_dim)
        x = torch.cat(
            [features, src_item_emb, tgt_item_emb, src_cat_emb, tgt_cat_emb, cat_l2_dist],
            dim=-1,
        )

        # Cross part
        x0 = x
        xl = x
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)
        cross_out = xl

        # Deep part
        deep_out = self.deep_layer(x)

        # Combine and predict
        combined = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.output(combined)
        return logits
