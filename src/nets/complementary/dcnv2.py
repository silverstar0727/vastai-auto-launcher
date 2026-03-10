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
        # x_l^T * w_l is a dot product per sample -> (batch, 1)
        interaction = torch.sum(xl * self.weight, dim=-1, keepdim=True)
        # x_0 * (interaction + b_l) + x_l
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
    """Deep & Cross Network V2 for complementary product prediction.

    Input:
        - features: (N, 20) behavioral features

    Output:
        - logits: (N, 1) raw logits for binary classification
    """

    def __init__(
        self,
        input_dim: int = 20,
        num_cross_layers: int = 3,
        deep_hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if deep_hidden_dims is None:
            deep_hidden_dims = [128, 64, 32, 16]

        self.input_dim = input_dim

        # Cross network
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layers)])

        # Deep network
        self.deep_layer = DeepLayer(input_dim, deep_hidden_dims, dropout)

        # Output: concat cross_output(input_dim) + deep_output(last_hidden) -> 1
        combined_dim = input_dim + self.deep_layer.output_dim
        self.output = nn.Linear(combined_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Cross part
        x0 = features
        xl = features
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)
        cross_out = xl  # (N, input_dim)

        # Deep part
        deep_out = self.deep_layer(features)  # (N, last_hidden)

        # Combine and predict
        combined = torch.cat([cross_out, deep_out], dim=-1)  # (N, input_dim + last_hidden)
        logits = self.output(combined)  # (N, 1)
        return logits
