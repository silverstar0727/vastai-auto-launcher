import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_LOOKUP_TABLE = "text_lookup"


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def _build_tower(input_size: int, num_hidden_layers: int, hidden_units: int) -> nn.Sequential:
    """Build a tower with Linear(bias=False) + BatchNorm + ELU hidden layers,
    ending with L2 normalization.

    For num_hidden_layers=1 with hidden_units=32:
        Linear(input_size, 64, bias=False) -> BN(64) -> ELU
        -> Linear(64, 32, bias=False) -> L2Norm
    """
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            in_dim = input_size
        else:
            in_dim = hidden_units * 2 ** (num_hidden_layers - i + 1)
        out_dim = hidden_units * 2 ** (num_hidden_layers - i)

        layers.append(nn.Linear(in_dim, out_dim, bias=False))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ELU())

    # Final projection to hidden_units with L2 norm
    layers.append(nn.Linear(hidden_units * 2, hidden_units, bias=False))
    layers.append(L2Norm())
    return nn.Sequential(*layers)


class TwoTowerNet(nn.Module):
    """Two-Tower network for Bandit pre-training.

    User tower: click history embeddings -> MLP -> L2 normalize -> user_emb (32d)
    Item tower: item features -> MLP -> L2 normalize -> item_emb (32d)
    Scoring: user_emb @ item_emb.T / temperature

    Compared to CBFNet, this version always L2-normalizes outputs and uses
    temperature-scaled scoring for softmax cross-entropy training.
    """

    def __init__(
        self,
        num_items: int,
        item_embed_size: int = 0,
        text_embed_size: int = 512,
        text_project_size: int = 128,
        n_markets: int = 0,
        market_embed_size: int = 128,
        n_categories: int = 0,
        category_embed_size: int = 64,
        use_age_info: bool = True,
        num_hidden_layers: int = 1,
        hidden_units: int = 32,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.num_items = num_items
        self.temperature = temperature
        self.use_age_info = use_age_info
        self.n_markets = n_markets
        self.n_categories = n_categories
        self.item_embed_size = item_embed_size

        vocab_size = num_items + 1  # 0 is padding

        # --- Shared embedding layers ---
        if item_embed_size > 0:
            self.item_embedding = nn.Embedding(vocab_size, item_embed_size, padding_idx=0)

        # Text: pretrained lookup (frozen) + linear projection
        self.register_buffer("text_lookup", torch.zeros(vocab_size, text_embed_size))
        self.text_project = nn.Linear(text_embed_size, text_project_size, bias=False)

        # Market / Category embeddings (shared between towers)
        if n_markets > 0:
            self.market_embedding = nn.Embedding(n_markets + 1, market_embed_size, padding_idx=0)
        if n_categories > 0:
            self.category_embedding = nn.Embedding(n_categories + 1, category_embed_size, padding_idx=0)

        # --- Compute input sizes ---
        # User tower: [item_embed] + text_project + [market] + [category] + [age(3)]
        user_input_size = text_project_size
        if item_embed_size > 0:
            user_input_size += item_embed_size
        if n_markets > 0:
            user_input_size += market_embed_size
        if n_categories > 0:
            user_input_size += category_embed_size
        if use_age_info:
            user_input_size += 3  # polynomial: 1, age, age^2

        # Item tower: text_project + [market] + [category]
        item_input_size = text_project_size
        if n_markets > 0:
            item_input_size += market_embed_size
        if n_categories > 0:
            item_input_size += category_embed_size

        # --- Towers (always L2-normalized) ---
        self.user_tower = _build_tower(user_input_size, num_hidden_layers, hidden_units)
        self.item_tower = _build_tower(item_input_size, num_hidden_layers, hidden_units)

        # --- Item feature buffers ---
        self.register_buffer("item_markets", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("item_categories", torch.zeros(vocab_size, dtype=torch.long))

    def forward(self, user_features, item_idxes=None):
        """
        Args:
            user_features: dict with keys click_items (N,T), and optionally
                click_markets (N,T), click_categories (N,T), user_age (N,1)
            item_idxes: (K,) subset of item indices to score against.
                If None, score against all items.

        Returns:
            scores: (N, K) or (N, num_items+1), temperature-scaled
        """
        user_emb = self.forward_user_tower(user_features)
        item_emb = self.forward_item_tower(item_idxes)
        scores = torch.matmul(user_emb, item_emb.T) / self.temperature
        return scores

    def forward_user_tower(self, user_features):
        """Compute user embeddings from click history. Returns L2-normalized (N, D)."""
        click_items = user_features["click_items"]  # (N, T)
        mask = (click_items > 0).unsqueeze(-1).float()  # (N, T, 1)

        parts = []

        # Item embedding (optional, item_embed_size=0 means text-only)
        if self.item_embed_size > 0:
            item_emb = self.item_embedding(click_items)  # (N, T, D)
            item_emb_pooled = (item_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            parts.append(item_emb_pooled)

        # Text embedding: mean-pool over sequence
        text_emb = self.text_project(
            self.text_lookup[click_items.cpu()].to(click_items.device)
        )  # (N, T, P)
        text_emb_pooled = (text_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        parts.append(text_emb_pooled)

        # Market embedding
        if self.n_markets > 0 and "click_markets" in user_features:
            market_emb = self.market_embedding(user_features["click_markets"])  # (N, T, M)
            market_pooled = (market_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            parts.append(market_pooled)

        # Category embedding
        if self.n_categories > 0 and "click_categories" in user_features:
            cat_emb = self.category_embedding(user_features["click_categories"])  # (N, T, C)
            cat_pooled = (cat_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            parts.append(cat_pooled)

        # Age polynomial features
        if self.use_age_info and "user_age" in user_features:
            age = user_features["user_age"].float()  # (N, 1)
            age_features = torch.cat([torch.ones_like(age), age, age ** 2], dim=-1)
            parts.append(age_features)

        user_input = torch.cat(parts, dim=-1)
        return self.user_tower(user_input)  # (N, D), L2-normalized

    def forward_item_tower(self, item_idxes=None):
        """Compute item embeddings. Returns L2-normalized (K, D)."""
        if item_idxes is not None:
            target_idxes = item_idxes.reshape(-1)
        else:
            target_idxes = torch.arange(
                0, self.num_items + 1, dtype=torch.long, device=self.text_lookup.device
            )

        parts = []

        # Text embedding (projected)
        text_emb = self.text_project(
            self.text_lookup[target_idxes.cpu()].to(target_idxes.device)
        )
        parts.append(text_emb)

        # Market embedding
        if self.n_markets > 0:
            market_emb = self.market_embedding(self.item_markets[target_idxes])
            parts.append(market_emb)

        # Category embedding
        if self.n_categories > 0:
            cat_emb = self.category_embedding(self.item_categories[target_idxes])
            parts.append(cat_emb)

        item_input = torch.cat(parts, dim=-1)
        return self.item_tower(item_input)  # (K, D), L2-normalized

    def get_all_item_embeddings(self):
        """Compute embeddings for all items (eval mode, no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward_item_tower(item_idxes=None)

    def get_all_user_embeddings(self, user_features):
        """Compute user embeddings (eval mode, no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward_user_tower(user_features)

    # --- State management for text/item features ---

    def set_text_embeddings(self, embed_dict):
        """Set pretrained text embeddings from a dict {item_index: vector}."""
        for item_index, vector in embed_dict.items():
            self.text_lookup[item_index] = torch.FloatTensor(vector)

    def get_text_embedding(self):
        return self.text_lookup.cpu().numpy().copy()

    def set_item_features(self, item_markets=None, item_categories=None):
        """Set per-item static features (market/category indices)."""
        if item_markets is not None:
            self.item_markets.copy_(torch.LongTensor(item_markets))
        if item_categories is not None:
            self.item_categories.copy_(torch.LongTensor(item_categories))

    def load_state_dict(self, state_dict, strict=True):
        if TEXT_LOOKUP_TABLE in state_dict:
            text_lookup = state_dict.pop(TEXT_LOOKUP_TABLE)
            self.text_lookup = torch.FloatTensor(text_lookup)
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)
        destination[TEXT_LOOKUP_TABLE] = self.get_text_embedding()
        return destination
