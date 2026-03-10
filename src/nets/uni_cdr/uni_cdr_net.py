import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """BERT-style transformer encoder for sequential user history."""

    def __init__(
        self,
        input_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        intermediate_dim: int = 1024,
        max_len: int = 50,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.position = nn.Embedding(max_len, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, D) embedded sequence
            mask: (N, T) boolean, True = valid position
        Returns:
            (N, D) pooled user representation
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position(positions)

        # nn.TransformerEncoder expects src_key_padding_mask where True = ignore
        padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # Mean-pool over valid positions
        mask_expanded = mask.unsqueeze(-1).float()  # (N, T, 1)
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled


class DNNTower(nn.Module):
    """DNN tower: Linear -> BN -> ELU layers with optional L2 normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.15,
        use_l2_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.use_l2_norm = use_l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.use_l2_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class UserTower(nn.Module):
    """User tower: embedding layers + transformer encoder + DNN head."""

    def __init__(
        self,
        num_items: int,
        item_embed_size: int = 256,
        text_embed_size: int = 512,
        market_embed_size: int = 128,
        category_embed_size: int = 64,
        num_markets: int = 16,
        num_categories: int = 512,
        transformer_input_len: int = 50,
        transformer_embed_size: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        intermediate_dim: int = 1024,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        dropout: float = 0.15,
        use_l2_norm: bool = True,
    ):
        super().__init__()
        self.transformer_input_len = transformer_input_len

        # Embedding layers
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_size, padding_idx=0)
        self.text_projection = nn.Linear(text_embed_size, item_embed_size, bias=False)
        self.market_embedding = nn.Embedding(num_markets, market_embed_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, category_embed_size, padding_idx=0)

        embed_dim = item_embed_size + item_embed_size + market_embed_size + category_embed_size
        self.input_proj = nn.Linear(embed_dim, transformer_embed_size)

        # Transformer
        self.transformer = TransformerEncoder(
            input_dim=transformer_embed_size,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            intermediate_dim=intermediate_dim,
            max_len=transformer_input_len,
            dropout=dropout,
        )

        # DNN head
        self.dnn = DNNTower(
            input_dim=transformer_embed_size,
            hidden_dim=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            use_l2_norm=use_l2_norm,
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        market_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (N, T) item index sequence
            text_embeds: (N, T, text_embed_size) pre-computed text embeddings
            market_ids: (N, T) market indices
            category_ids: (N, T) category indices
        Returns:
            (N, hidden_dim) user embedding
        """
        # Truncate to transformer_input_len (take last T items)
        T = self.transformer_input_len
        item_ids = item_ids[:, -T:]
        text_embeds = text_embeds[:, -T:]
        market_ids = market_ids[:, -T:]
        category_ids = category_ids[:, -T:]

        item_emb = self.item_embedding(item_ids)
        text_emb = self.text_projection(text_embeds)
        market_emb = self.market_embedding(market_ids)
        cat_emb = self.category_embedding(category_ids)

        x = torch.cat([item_emb, text_emb, market_emb, cat_emb], dim=-1)
        x = self.input_proj(x)

        mask = item_ids > 0  # (N, T)
        x = self.transformer(x, mask)
        return self.dnn(x)


class ItemTower(nn.Module):
    """Item tower: embedding lookup + DNN head."""

    def __init__(
        self,
        num_items: int,
        item_embed_size: int = 256,
        text_embed_size: int = 512,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        dropout: float = 0.15,
        use_l2_norm: bool = True,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_size, padding_idx=0)
        self.text_projection = nn.Linear(text_embed_size, item_embed_size, bias=False)

        self.dnn = DNNTower(
            input_dim=item_embed_size + item_embed_size,
            hidden_dim=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            use_l2_norm=use_l2_norm,
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (N, K) candidate item indices
            text_embeds: (N, K, text_embed_size) pre-computed text embeddings
        Returns:
            (N, K, hidden_dim) item embeddings
        """
        item_emb = self.item_embedding(item_ids)
        text_emb = self.text_projection(text_embeds)
        x = torch.cat([item_emb, text_emb], dim=-1)

        # DNNTower expects 2D input; reshape
        N, K, D = x.shape
        x = x.view(N * K, D)
        x = self.dnn(x)
        return x.view(N, K, -1)


# ---------------------------------------------------------------------------
# Full UniCDR network
# ---------------------------------------------------------------------------


class UniCDRNet(nn.Module):
    """Unified Cross-Domain Recommendation network.

    Two domains (source and target) each have user and item towers.
    Transfer matrices map user representations between domains.
    Produces 6 score types for multi-objective training.

    Input dict per domain:
        - item_ids:      (N, T) user click history
        - text_embeds:   (N, T, text_embed_size) text features
        - market_ids:    (N, T) market indices
        - category_ids:  (N, T) category indices
        - cand_item_ids: (N, K) candidate item indices
        - cand_text_embeds: (N, K, text_embed_size) candidate text features
        - n_interactions: (N,) number of interactions in this domain

    Output dict:
        - src_scores, target_scores: within-domain scores
        - src_by_shared, target_by_shared: shared user scores
        - src_by_masking, target_by_masking: cross-domain scores
        - src_user_emb, target_user_emb, shared_user_emb: user embeddings
    """

    def __init__(
        self,
        src_num_items: int,
        target_num_items: int,
        item_embed_size: int = 256,
        text_embed_size: int = 512,
        market_embed_size: int = 128,
        category_embed_size: int = 64,
        num_markets: int = 16,
        num_categories: int = 512,
        transformer_input_len: int = 50,
        transformer_embed_size: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        intermediate_dim: int = 1024,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        dropout: float = 0.15,
        use_l2_norm: bool = True,
        softmax_temperature: float = 0.05,
    ):
        super().__init__()
        self.src_num_items = src_num_items
        self.target_num_items = target_num_items
        self.softmax_temperature = softmax_temperature

        user_tower_kwargs = dict(
            item_embed_size=item_embed_size,
            text_embed_size=text_embed_size,
            market_embed_size=market_embed_size,
            category_embed_size=category_embed_size,
            num_markets=num_markets,
            num_categories=num_categories,
            transformer_input_len=transformer_input_len,
            transformer_embed_size=transformer_embed_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            intermediate_dim=intermediate_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_l2_norm=use_l2_norm,
        )

        item_tower_kwargs = dict(
            item_embed_size=item_embed_size,
            text_embed_size=text_embed_size,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_l2_norm=use_l2_norm,
        )

        # Source towers
        self.src_user_tower = UserTower(num_items=src_num_items, **user_tower_kwargs)
        self.src_item_tower = ItemTower(num_items=src_num_items, **item_tower_kwargs)

        # Target towers
        self.target_user_tower = UserTower(num_items=target_num_items, **user_tower_kwargs)
        self.target_item_tower = ItemTower(num_items=target_num_items, **item_tower_kwargs)

        # Transfer matrices for cross-domain mapping
        self.transfer_src_to_target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transfer_target_to_src = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        src_inputs: dict[str, torch.Tensor],
        target_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # --- User embeddings ---
        src_user_emb = self.src_user_tower(
            src_inputs["item_ids"],
            src_inputs["text_embeds"],
            src_inputs["market_ids"],
            src_inputs["category_ids"],
        )  # (N, D)

        target_user_emb = self.target_user_tower(
            target_inputs["item_ids"],
            target_inputs["text_embeds"],
            target_inputs["market_ids"],
            target_inputs["category_ids"],
        )  # (N, D)

        # --- Shared user embedding (weighted average by interaction counts) ---
        n_src = src_inputs["n_interactions"].float().unsqueeze(-1)  # (N, 1)
        n_target = target_inputs["n_interactions"].float().unsqueeze(-1)
        total = (n_src + n_target).clamp(min=1)
        shared_user_emb = (n_src * src_user_emb + n_target * target_user_emb) / total

        # --- Item embeddings ---
        src_item_emb = self.src_item_tower(
            src_inputs["cand_item_ids"],
            src_inputs["cand_text_embeds"],
        )  # (N, K_src, D)

        target_item_emb = self.target_item_tower(
            target_inputs["cand_item_ids"],
            target_inputs["cand_text_embeds"],
        )  # (N, K_target, D)

        # --- 6 score types ---
        t = self.softmax_temperature

        # 1-2: Within-domain scores
        src_scores = torch.bmm(src_user_emb.unsqueeze(1), src_item_emb.transpose(1, 2)).squeeze(1) / t
        target_scores = torch.bmm(target_user_emb.unsqueeze(1), target_item_emb.transpose(1, 2)).squeeze(1) / t

        # 3-4: Shared user scores
        src_by_shared = torch.bmm(shared_user_emb.unsqueeze(1), src_item_emb.transpose(1, 2)).squeeze(1) / t
        target_by_shared = torch.bmm(shared_user_emb.unsqueeze(1), target_item_emb.transpose(1, 2)).squeeze(1) / t

        # 5-6: Cross-domain (masking) scores
        target_user_to_src = self.transfer_target_to_src(target_user_emb)
        src_user_to_target = self.transfer_src_to_target(src_user_emb)

        src_by_masking = torch.bmm(target_user_to_src.unsqueeze(1), src_item_emb.transpose(1, 2)).squeeze(1) / t
        target_by_masking = torch.bmm(src_user_to_target.unsqueeze(1), target_item_emb.transpose(1, 2)).squeeze(1) / t

        return {
            "src_scores": src_scores,
            "target_scores": target_scores,
            "src_by_shared": src_by_shared,
            "target_by_shared": target_by_shared,
            "src_by_masking": src_by_masking,
            "target_by_masking": target_by_masking,
            "src_user_emb": src_user_emb,
            "target_user_emb": target_user_emb,
            "shared_user_emb": shared_user_emb,
        }
