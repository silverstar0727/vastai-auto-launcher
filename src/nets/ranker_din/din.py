import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalActivationUnit(nn.Module):
    """Local activation unit for DIN attention score computation.

    Computes attention scores by concatenating [query, key, query-key, query*key]
    and passing through a two-layer MLP.

    Reference: https://arxiv.org/pdf/1706.06978.pdf
    """

    def __init__(self, hidden_units: int = 36, embedding_dim: int = 256):
        super().__init__()
        self.dnn = nn.Linear(
            in_features=4 * embedding_dim,
            out_features=hidden_units,
        )
        self.activation = nn.ELU()
        self.dense = nn.Linear(hidden_units, 1)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores.

        Args:
            query: (B, 1, D) candidate item embedding.
            keys: (B, T, D) user behavior sequence embeddings.

        Returns:
            attention_score: (B, T, 1)
        """
        seq_len = keys.size(1)
        queries = query.expand(-1, seq_len, -1)  # (B, T, D)

        attention_input = torch.cat(
            [queries, keys, queries - keys, queries * keys],
            dim=-1,
        )  # (B, T, 4*D)
        attention_output = self.activation(self.dnn(attention_input))
        attention_score = self.dense(attention_output)  # (B, T, 1)
        return attention_score


class AttentionSequencePoolingLayer(nn.Module):
    """DIN Attention Sequence Pooling Layer.

    Applies local activation unit to compute attention weights over a key sequence,
    masks out padding positions, normalizes via softmax, and produces a weighted sum.

    Reference: https://arxiv.org/pdf/1706.06978.pdf
    """

    def __init__(
        self,
        att_hidden_units: int = 36,
        weight_normalization: bool = True,
        embedding_dim: int = 4,
        projection_dim: int = 0,
    ):
        super().__init__()
        self.weight_normalization = weight_normalization

        if projection_dim > 0:
            self.projection_layer = nn.Linear(embedding_dim, projection_dim, bias=False)
            self.local_att = LocalActivationUnit(
                hidden_units=att_hidden_units,
                embedding_dim=projection_dim,
            )
        else:
            self.projection_layer = None
            self.local_att = LocalActivationUnit(
                hidden_units=att_hidden_units,
                embedding_dim=embedding_dim,
            )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        keys_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-weighted pooling of key sequence.

        Args:
            query: (B, 1, D) candidate item embedding.
            keys: (B, T, D) user behavior sequence embeddings.
            keys_mask: (B, T) boolean mask, True for valid positions.

        Returns:
            output: (B, 1, D) attention-pooled representation.
        """
        if self.projection_layer is not None:
            attention_score = self.local_att(
                self.projection_layer(query), self.projection_layer(keys)
            )  # (B, T, 1)
        else:
            attention_score = self.local_att(query, keys)  # (B, T, 1)

        outputs = attention_score.transpose(1, 2)  # (B, 1, T)

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_mask.unsqueeze(1), outputs, paddings)  # (B, 1, T)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # (B, 1, T)

        outputs = torch.matmul(outputs, keys)  # (B, 1, D)
        return outputs
