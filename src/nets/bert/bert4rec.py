import numpy as np
import torch
import torch.nn as nn

from .layers import BERTEmbedding, TextEmbedding, TimeInterval, TransformerBlock


class BERT4Rec(nn.Module):
    """BERT4Rec network architecture.

    Input features:
        - item_indexes: (N, T) sequence of item indices
        - events: (N, T) sequence of event codes
        - time_intervals: (N, T) normalized time intervals

    Output:
        - logits: (N, T, num_items+1)
    """

    def __init__(
        self,
        num_items: int,
        max_len: int = 80,
        num_blocks: int = 2,
        num_heads: int = 4,
        hidden_units: int = 256,
        item_embed_size: int = 64,
        dropout: float = 0.1,
        text_embed_size: int = 512,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.hidden_units = hidden_units

        self.time_interval_layer = TimeInterval()
        text_vsize = hidden_units - self.time_interval_layer.get_num_features() - item_embed_size

        # text_lookup is initialized to zeros; set via set_text_embeddings() after init
        text_lookup = np.zeros((num_items + 2, text_embed_size))
        self.text_embedding = TextEmbedding(text_lookup, text_vsize)

        self.bert_embedding = BERTEmbedding(
            vocab_size=num_items + 2,
            embed_size=item_embed_size,
            max_len=max_len,
            dropout=0.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_units, num_heads, hidden_units * 4, dropout) for _ in range(num_blocks)]
        )

        self.out = nn.Linear(hidden_units, num_items + 1)

    @property
    def mask_index(self):
        return self.num_items + 1

    def forward(self, item_indexes, events, time_intervals):
        mask = self._get_mask(item_indexes)

        item_embeddings = self.bert_embedding(item_indexes, events)
        text_embeddings = self.text_embedding(item_indexes)
        time_features = self.time_interval_layer(time_intervals)
        x = torch.cat((item_embeddings, text_embeddings, time_features), dim=-1)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.out(x)

    def set_text_embeddings(self, embed_dict):
        for item_index, vector in embed_dict.items():
            self.text_embedding.text_lookup[item_index] = torch.FloatTensor(vector)

    def get_text_embedding(self):
        return self.text_embedding.text_lookup.detach().cpu().numpy().copy()

    @staticmethod
    def _get_mask(item_indexes):
        n_times = item_indexes.size(1)
        mask = (item_indexes > 0).unsqueeze(1)  # (N, 1, T)
        mask = mask.repeat(1, n_times, 1)  # (N, T, T)
        mask = mask.unsqueeze(1)  # (N, 1, T, T)
        return mask
