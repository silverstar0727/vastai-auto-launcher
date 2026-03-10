import math

import torch
import torch.nn as nn


class DualTower(nn.Module):
    """Dual-tower scoring head: separate user/item towers with dot-product scoring."""

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_units: int = 256,
        hidden_layers: int = 1,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_tower = self._build_tower(user_input_dim, hidden_units, hidden_layers, output_dim, dropout)
        self.item_tower = self._build_tower(item_input_dim, hidden_units, hidden_layers, output_dim, dropout)

    def forward(self, user_emb, item_emb):
        """
        Args:
            user_emb: (batch, user_input_dim)
            item_emb: (batch, num_items, item_input_dim) or (batch, item_input_dim)
        Returns:
            scores: (batch, num_items) or (batch,)
        """
        u = self.user_tower(user_emb)  # (batch, output_dim)
        if item_emb.dim() == 3:
            i = self.item_tower(item_emb)  # (batch, num_items, output_dim)
            # (batch, 1, output_dim) @ (batch, output_dim, num_items) -> (batch, 1, num_items) -> (batch, num_items)
            scores = torch.bmm(u.unsqueeze(1), i.transpose(1, 2)).squeeze(1)
        else:
            i = self.item_tower(item_emb)  # (batch, output_dim)
            scores = (u * i).sum(dim=-1)  # (batch,)
        return scores

    @staticmethod
    def _build_tower(input_dim, hidden_units, hidden_layers, output_dim, dropout):
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)


class SearchRankerNet(nn.Module):
    """Multi-task search ranker with transformer encoder and dual-tower task heads.

    Architecture:
    - Embedding layers: item, text (query tokens), market, category
    - Transformer encoder (BERT-style) for user click sequence + query
    - 4 dual-tower heads: retriever, ctr, ctcar, ctcvr
    - Retriever uses softmax ranking; ctr/ctcar/ctcvr use BCE

    Derived probabilities:
        CTCAR = p_ctr * p_click_to_action
        CTCVR = p_ctcar * p_action_to_order
    """

    TASK_NAMES = ("retriever", "ctr", "ctcar", "ctcvr")

    def __init__(
        self,
        num_items: int = 260000,
        num_markets: int = 100,
        num_categories: int = 5000,
        vocab_size: int = 30000,
        item_embed_size: int = 256,
        market_embed_size: int = 128,
        category_embed_size: int = 64,
        text_embed_size: int = 64,
        transformer_input_len: int = 80,
        transformer_embed_size: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_intermediate: int = 1024,
        hidden_units: int = 256,
        hidden_layers: int = 1,
        tower_output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.transformer_input_len = transformer_input_len
        self.transformer_embed_size = transformer_embed_size

        # --- Embedding layers ---
        self.item_embedding = nn.Embedding(num_items + 2, item_embed_size, padding_idx=0)
        # +2: 0=padding, num_items+1=special query token
        self.market_embedding = nn.Embedding(num_markets + 1, market_embed_size, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, category_embed_size, padding_idx=0)
        self.text_embedding = nn.Embedding(vocab_size + 1, text_embed_size, padding_idx=0)

        # --- Sequence projector: concat(item, market, category) -> transformer_embed_size ---
        seq_input_dim = item_embed_size + market_embed_size + category_embed_size  # 448
        self.seq_projector = nn.Sequential(
            nn.Linear(seq_input_dim, transformer_embed_size),
            nn.BatchNorm1d(transformer_embed_size),
            nn.ELU(),
        )

        # --- Position embedding for transformer input ---
        # +1 for the special query token prepended to the sequence
        self.position_embedding = nn.Embedding(transformer_input_len + 1, transformer_embed_size)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_size,
            nhead=transformer_heads,
            dim_feedforward=transformer_intermediate,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.transformer_norm = nn.LayerNorm(transformer_embed_size)

        # --- Query projector: text embedding -> transformer_embed_size ---
        self.query_projector = nn.Sequential(
            nn.Linear(text_embed_size, transformer_embed_size),
            nn.LayerNorm(transformer_embed_size),
            nn.ELU(),
        )

        # --- User representation dim = transformer output ---
        user_dim = transformer_embed_size

        # --- Item feature dim for towers ---
        item_feature_dim = item_embed_size

        # --- 4 dual-tower task heads ---
        self.tower_retriever = DualTower(
            user_input_dim=user_dim,
            item_input_dim=item_feature_dim,
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
            output_dim=tower_output_dim,
            dropout=dropout,
        )
        self.tower_ctr = DualTower(
            user_input_dim=user_dim,
            item_input_dim=item_feature_dim,
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
            output_dim=tower_output_dim,
            dropout=dropout,
        )
        self.tower_ctcar = DualTower(
            user_input_dim=user_dim,
            item_input_dim=item_feature_dim,
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
            output_dim=tower_output_dim,
            dropout=dropout,
        )
        self.tower_ctcvr = DualTower(
            user_input_dim=user_dim,
            item_input_dim=item_feature_dim,
            hidden_units=hidden_units,
            hidden_layers=hidden_layers,
            output_dim=tower_output_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        click_items: torch.Tensor,
        click_markets: torch.Tensor,
        click_categories: torch.Tensor,
        query_tokens: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> dict:
        """
        Args:
            click_items: (batch, seq_len) item IDs from click history
            click_markets: (batch, seq_len) market IDs per clicked item
            click_categories: (batch, seq_len) category IDs per clicked item
            query_tokens: (batch, query_len) tokenized last search query
            candidate_items: (batch, num_candidates) candidate item IDs for scoring

        Returns:
            dict with keys "retriever", "ctr", "ctcar", "ctcvr",
            each (batch, num_candidates) logits/scores.
        """
        # --- Build user representation via transformer ---
        user_emb = self._encode_user(click_items, click_markets, click_categories, query_tokens)

        # --- Candidate item embeddings ---
        item_emb = self.item_embedding(candidate_items)  # (batch, num_candidates, item_embed_size)

        # --- Score via dual towers ---
        return {
            "retriever": self.tower_retriever(user_emb, item_emb),
            "ctr": self.tower_ctr(user_emb, item_emb),
            "ctcar": self.tower_ctcar(user_emb, item_emb),
            "ctcvr": self.tower_ctcvr(user_emb, item_emb),
        }

    def _encode_user(self, click_items, click_markets, click_categories, query_tokens):
        """Encode user from click history + search query via transformer.

        Returns:
            user_emb: (batch, transformer_embed_size)
        """
        batch_size = click_items.size(0)

        # Truncate sequence to transformer_input_len (take the most recent)
        click_items = click_items[:, -self.transformer_input_len:]
        click_markets = click_markets[:, -self.transformer_input_len:]
        click_categories = click_categories[:, -self.transformer_input_len:]

        # Embed click sequence
        item_emb = self.item_embedding(click_items)        # (batch, seq, item_embed_size)
        market_emb = self.market_embedding(click_markets)   # (batch, seq, market_embed_size)
        cat_emb = self.category_embedding(click_categories) # (batch, seq, category_embed_size)

        # Concat and project: (batch, seq, 448) -> (batch, seq, 256)
        seq_concat = torch.cat([item_emb, market_emb, cat_emb], dim=-1)
        seq_len = seq_concat.size(1)
        seq_flat = seq_concat.view(-1, seq_concat.size(-1))
        seq_proj = self.seq_projector(seq_flat).view(batch_size, seq_len, -1)

        # Build query token: mean-pool text embeddings -> project
        query_emb = self.text_embedding(query_tokens)  # (batch, query_len, text_embed_size)
        query_mask = (query_tokens != 0).unsqueeze(-1).float()
        query_lengths = query_mask.sum(dim=1).clamp(min=1.0)
        query_pooled = (query_emb * query_mask).sum(dim=1) / query_lengths  # (batch, text_embed_size)
        query_proj = self.query_projector(query_pooled)  # (batch, transformer_embed_size)

        # Prepend query token to sequence: [query, item1, item2, ..., itemN]
        query_proj = query_proj.unsqueeze(1)  # (batch, 1, transformer_embed_size)
        transformer_input = torch.cat([query_proj, seq_proj], dim=1)  # (batch, seq+1, 256)

        # Add position embeddings
        total_len = transformer_input.size(1)
        positions = torch.arange(total_len, device=transformer_input.device).unsqueeze(0)
        transformer_input = transformer_input + self.position_embedding(positions)

        # Build padding mask: padding where click_items == 0
        # query token (position 0) is never masked
        click_pad_mask = click_items == 0  # (batch, seq)
        query_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=click_items.device)
        src_key_padding_mask = torch.cat([query_pad, click_pad_mask], dim=1)  # (batch, seq+1)

        # Transformer encode
        output = self.transformer(transformer_input, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_norm(output)

        # Take the query token output (position 0) as user representation
        user_emb = output[:, 0, :]  # (batch, transformer_embed_size)
        return user_emb

    def set_dropout(self, dropout: float):
        """Update dropout rate for all modules."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
