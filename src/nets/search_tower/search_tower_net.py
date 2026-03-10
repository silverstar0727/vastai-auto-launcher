import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TextEmbedding(nn.Module):
    """Projects pretrained text embeddings to target dimension."""

    def __init__(self, text_lookup, hidden_units):
        super().__init__()
        self.text_lookup = torch.FloatTensor(text_lookup)
        self.linear = nn.Linear(text_lookup.shape[-1], hidden_units, bias=False)
        self.activation = nn.ReLU()
        self.norm = LayerNorm(hidden_units)

    def forward(self, item_indexes):
        embed_vectors = self.text_lookup[item_indexes.cpu()]
        embed_vectors = embed_vectors.to(item_indexes.device)
        return self.norm(self.activation(self.linear(embed_vectors)))


class QueryTimeFeatures(nn.Module):
    """Polynomial time features: (t, t^2, t^3, sqrt(t)) -> 4 features."""

    N_FEATURES = 4

    def forward(self, time_steps):
        t = time_steps.unsqueeze(-1)  # (B, 1)
        return torch.cat((t, t**2, t**3, t**0.5), dim=-1)  # (B, 4)

    def get_num_features(self):
        return self.N_FEATURES


TEXT_LOOKUP_TABLE = "text_lookup"


class SearchTowerNet(nn.Module):
    """Search Tower network architecture.

    Predicts items for a given search query + user click history.

    Input features:
        - click_items: (B, max_len) sequence of item indices from click history
        - query_tokens: (B, max_query_len) token indices for the search query
        - query_time: (B,) normalized timestamp of the query

    Output:
        - logits: (B, num_items+1)
    """

    def __init__(
        self,
        num_items: int,
        item_embed_size: int = 128,
        hidden_units: int = 256,
        max_len: int = 80,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        text_embed_size: int = 512,
        query_token_embed_size: int = 256,
        max_query_len: int = 32,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_units = hidden_units

        # Item embedding (vocab includes padding=0, items 1..num_items, no mask token needed)
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_size, padding_idx=0)

        # Pretrained text embedding projected to fill remaining hidden dims
        text_proj_size = hidden_units - item_embed_size
        text_lookup = np.zeros((num_items + 1, text_embed_size))
        self.text_embedding = TextEmbedding(text_lookup, text_proj_size)

        # Positional embedding: max_len items + 1 query position
        self.position_embedding = nn.Embedding(max_len + 1, hidden_units)

        # Query token embedding (pretrained token embeddings projected to hidden)
        self.query_token_embedding = nn.Embedding(30000, query_token_embed_size, padding_idx=0)
        self.query_projection = nn.Linear(query_token_embed_size, hidden_units)

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_units, num_heads, hidden_units * 4, dropout) for _ in range(num_blocks)]
        )

        # Query time features
        self.time_features = QueryTimeFeatures()
        time_feat_size = self.time_features.get_num_features()

        # Output head: transformer output (hidden_units) + time features -> logits
        output_input_size = hidden_units + time_feat_size
        self.output_hidden = nn.Linear(output_input_size, hidden_units)
        self.output_bn = nn.BatchNorm1d(hidden_units)
        self.output_act = nn.ELU()
        self.output_head = nn.Linear(hidden_units, num_items + 1)

    def forward(self, click_items, query_tokens, query_time):
        """
        Args:
            click_items: (B, max_len) item indices from click history
            query_tokens: (B, max_query_len) token indices for search query
            query_time: (B,) normalized query timestamps
        Returns:
            logits: (B, num_items+1)
        """
        batch_size = click_items.size(0)
        device = click_items.device

        # --- Encode click history items ---
        # Item embeddings + text projection -> (B, max_len, hidden_units)
        item_emb = self.item_embedding(click_items)  # (B, max_len, item_embed_size)
        text_emb = self.text_embedding(click_items)  # (B, max_len, text_proj_size)
        click_emb = torch.cat([item_emb, text_emb], dim=-1)  # (B, max_len, hidden_units)

        # --- Encode query as a single token ---
        q_tok_emb = self.query_token_embedding(query_tokens)  # (B, max_query_len, query_token_embed_size)
        q_pooled = q_tok_emb.mean(dim=1)  # (B, query_token_embed_size)
        q_projected = self.query_projection(q_pooled).unsqueeze(1)  # (B, 1, hidden_units)

        # --- Build transformer input: [click_items..., query_token] ---
        seq = torch.cat([click_emb, q_projected], dim=1)  # (B, max_len+1, hidden_units)

        # Add positional embeddings
        positions = torch.arange(self.max_len + 1, device=device).unsqueeze(0)  # (1, max_len+1)
        seq = seq + self.position_embedding(positions)

        # Build attention mask (mask padding positions in click history)
        # query position (last) is always valid
        click_mask = (click_items > 0)  # (B, max_len)
        query_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        valid_mask = torch.cat([click_mask, query_valid], dim=1)  # (B, max_len+1)
        seq_len = self.max_len + 1
        attn_mask = valid_mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, seq_len, -1)  # (B, 1, T, T)

        # Transformer
        x = seq
        for block in self.transformer_blocks:
            x = block(x, attn_mask)

        # Take the output at the query position (last position)
        query_output = x[:, -1, :]  # (B, hidden_units)

        # --- Time features ---
        time_feat = self.time_features(query_time)  # (B, 4)

        # --- Output head ---
        combined = torch.cat([query_output, time_feat], dim=-1)  # (B, hidden_units+4)
        h = self.output_hidden(combined)  # (B, hidden_units)
        h = self.output_bn(h)
        h = self.output_act(h)
        logits = self.output_head(h)  # (B, num_items+1)

        return logits

    def set_text_embeddings(self, embed_dict):
        """Set pretrained text embeddings for items."""
        for item_index, vector in embed_dict.items():
            self.text_embedding.text_lookup[item_index] = torch.FloatTensor(vector)

    def get_text_embedding(self):
        return self.text_embedding.text_lookup.numpy().copy()

    def load_state_dict(self, state_dict, strict=True):
        if TEXT_LOOKUP_TABLE in state_dict:
            text_lookup = state_dict.pop(TEXT_LOOKUP_TABLE)
            self.text_embedding.text_lookup = torch.FloatTensor(text_lookup)
        super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)
        destination[TEXT_LOOKUP_TABLE] = self.get_text_embedding()
        return destination
