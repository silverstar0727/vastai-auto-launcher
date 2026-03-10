from typing import Optional

import torch
import torch.nn as nn

TEXT_LOOKUP_TABLE = "text_lookup"


def _build_tower(input_size, num_hidden_layers, last_hidden_units):
    """Build a tower: stacked Linear(bias=False)+BN+ELU layers, then a final Linear."""
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            in_dim = input_size
        else:
            in_dim = last_hidden_units * 2 ** (num_hidden_layers - i + 1)
        out_dim = last_hidden_units * 2 ** (num_hidden_layers - i)

        layers.append(nn.Linear(in_dim, out_dim, bias=False))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ELU())

    layers.append(nn.Linear(last_hidden_units * 2, last_hidden_units, bias=False))
    return nn.Sequential(*layers)


class ComplementNet(nn.Module):
    """Two-Tower network for complement recommendations.

    Same architecture as CBFNet but with:
    - Item embedding included in both towers (use_item_emb_in_item_tower=True)
    - DropoutNet on item tower embedding
    - Smaller market/category embedding sizes (64 vs 128)

    User tower: item_emb + text_project + [market] + [category] -> hidden -> user_emb
    Item tower: text_project + [item_emb] + [market] + [category] -> hidden -> item_emb
    Scoring: user_emb @ item_emb.T

    Input features (provided as dicts by the DataModule):
        User tower:
            - click_items: (N, T) item index sequence

        Item tower (indexed via item_idxes or computed for all items):
            - item text embedding (pretrained, projected)
            - item embedding (with optional DropoutNet)
            - item market index   [optional]
            - item category index [optional]

    Output:
        - scores: (N, num_items+1) or (N, len(item_idxes))
    """

    def __init__(
        self,
        num_items: int,
        item_embed_size: int = 256,
        text_embed_size: int = 512,
        text_project_size: int = 128,
        n_markets: int = 0,
        market_embed_size: int = 64,
        n_categories: int = 0,
        category_embed_size: int = 64,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        item_dropout_prob: float = 0.0,
        use_item_emb_in_item_tower: bool = True,
    ):
        super().__init__()
        self.num_items = num_items
        self.item_dropout_prob = item_dropout_prob
        self.use_item_emb_in_item_tower = use_item_emb_in_item_tower
        self.n_markets = n_markets
        self.n_categories = n_categories

        # --- Shared embedding layers ---
        vocab_size = num_items + 1  # 0 is padding
        self.item_embedding = nn.Embedding(vocab_size, item_embed_size, padding_idx=0)

        # Text: pretrained lookup (not a parameter) + linear projection
        # text_lookup is initialized to zeros; call set_text_embeddings() after init
        self.register_buffer("text_lookup", torch.zeros(vocab_size, text_embed_size))
        self.text_project = nn.Linear(text_embed_size, text_project_size, bias=False)

        # Market / Category embeddings (shared between towers)
        if n_markets > 0:
            self.market_embedding = nn.Embedding(n_markets + 1, market_embed_size, padding_idx=0)
        if n_categories > 0:
            self.category_embedding = nn.Embedding(n_categories + 1, category_embed_size, padding_idx=0)

        # --- Compute input sizes ---
        # User tower input = item_embed + text_project + [market] + [category]
        user_input_size = item_embed_size + text_project_size
        if n_markets > 0:
            user_input_size += market_embed_size
        if n_categories > 0:
            user_input_size += category_embed_size

        # Item tower input = text_project + [item_embed] + [market] + [category]
        item_input_size = text_project_size
        if use_item_emb_in_item_tower:
            item_input_size += item_embed_size
        if n_markets > 0:
            item_input_size += market_embed_size
        if n_categories > 0:
            item_input_size += category_embed_size

        # --- Towers ---
        self.user_tower = _build_tower(user_input_size, num_hidden_layers, last_hidden_units)
        self.item_tower = _build_tower(item_input_size, num_hidden_layers, last_hidden_units)

        # --- Item feature buffers (set by DataModule via set_item_features) ---
        self.register_buffer("item_markets", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("item_categories", torch.zeros(vocab_size, dtype=torch.long))

    def forward(self, user_features, item_idxes=None):
        """
        Args:
            user_features: dict with keys click_items (N,T)
            item_idxes: (K,) subset of item indices to score against.
                If None, score against all items.

        Returns:
            scores: (N, K) or (N, num_items+1)
        """
        user_emb = self._forward_user_tower(user_features)
        item_emb = self._forward_item_tower(item_idxes)
        scores = torch.matmul(user_emb, item_emb.T)
        return scores

    def _forward_user_tower(self, user_features):
        click_items = user_features["click_items"]  # (N, T)

        parts = []

        # Item embedding: mean-pool over sequence (ignoring padding)
        item_emb = self.item_embedding(click_items)  # (N, T, D)
        mask = (click_items > 0).unsqueeze(-1).float()  # (N, T, 1)
        item_emb_pooled = (item_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (N, D)
        parts.append(item_emb_pooled)

        # Text embedding: mean-pool over sequence (projected)
        text_emb = self.text_project(self.text_lookup[click_items.cpu()].to(click_items.device))  # (N, T, P)
        text_emb_pooled = (text_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (N, P)
        parts.append(text_emb_pooled)

        # Market embedding
        if self.n_markets > 0:
            market_ids = self.item_markets[click_items]  # (N, T)
            market_emb = self.market_embedding(market_ids)  # (N, T, M)
            market_pooled = (market_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            parts.append(market_pooled)

        # Category embedding
        if self.n_categories > 0:
            cat_ids = self.item_categories[click_items]  # (N, T)
            cat_emb = self.category_embedding(cat_ids)  # (N, T, C)
            cat_pooled = (cat_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            parts.append(cat_pooled)

        user_input = torch.cat(parts, dim=-1)  # (N, user_input_size)
        return self.user_tower(user_input)

    def _forward_item_tower(self, item_idxes=None):
        if item_idxes is not None:
            target_idxes = item_idxes.reshape(-1)
        else:
            target_idxes = torch.arange(0, self.num_items + 1, dtype=torch.long, device=self.text_lookup.device)

        parts = []

        # Text embedding (projected)
        text_emb = self.text_project(self.text_lookup[target_idxes.cpu()].to(target_idxes.device))  # (K, P)
        parts.append(text_emb)

        # Item embedding (with optional DropoutNet)
        if self.use_item_emb_in_item_tower:
            if self.item_dropout_prob > 0 and self.training:
                preserve_mask = torch.empty_like(target_idxes, dtype=torch.float).bernoulli_(
                    1.0 - self.item_dropout_prob
                )
                dropped_idxes = (target_idxes.float() * preserve_mask).long()
            else:
                dropped_idxes = target_idxes
            item_emb = self.item_embedding(dropped_idxes)  # (K, D)
            parts.append(item_emb)

        # Market embedding
        if self.n_markets > 0:
            market_emb = self.market_embedding(self.item_markets[target_idxes])  # (K, M)
            parts.append(market_emb)

        # Category embedding
        if self.n_categories > 0:
            cat_emb = self.category_embedding(self.item_categories[target_idxes])  # (K, C)
            parts.append(cat_emb)

        item_input = torch.cat(parts, dim=-1)  # (K, item_input_size)
        return self.item_tower(item_input)

    def get_item_tower_embed(self):
        """Compute embeddings for all items. Used for evaluation / export."""
        self.eval()
        with torch.no_grad():
            return self._forward_item_tower(item_idxes=None)

    def set_text_embeddings(self, embed_dict):
        """Set pretrained text embeddings from a dict {item_index: vector}."""
        for item_index, vector in embed_dict.items():
            self.text_lookup[item_index] = torch.FloatTensor(vector)

    def get_text_embedding(self):
        return self.text_lookup.cpu().numpy().copy()

    def set_item_features(self, item_markets=None, item_categories=None):
        """Set per-item static features (market/category indices)."""
        if item_markets is not None:
            t = torch.LongTensor(item_markets)
            n = min(len(t), len(self.item_markets))
            self.item_markets[:n] = t[:n]
        if item_categories is not None:
            t = torch.LongTensor(item_categories)
            n = min(len(t), len(self.item_categories))
            self.item_categories[:n] = t[:n]

    def load_state_dict(self, state_dict, strict=True):
        if TEXT_LOOKUP_TABLE in state_dict:
            text_lookup = state_dict.pop(TEXT_LOOKUP_TABLE)
            self.text_lookup = torch.FloatTensor(text_lookup)
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)
        destination[TEXT_LOOKUP_TABLE] = self.get_text_embedding()
        return destination
