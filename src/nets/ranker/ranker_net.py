import torch
import torch.nn as nn


class CrossNet(nn.Module):
    """Explicit feature crossing network (DCN-style)."""

    def __init__(self, input_dim: int, layer_num: int = 2):
        super().__init__()
        self.layer_num = layer_num
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, 1)) for _ in range(layer_num)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(layer_num)]
        )
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.layer_num):
            # x_{i+1} = x0 * (xi^T * w_i) + b_i + xi
            cross = x0 * torch.matmul(xi, self.weights[i])  # (batch, dim)
            xi = cross + self.biases[i] + xi
        return xi


class Tower(nn.Module):
    """Single task tower: MLP + CrossNet + output head."""

    def __init__(
        self,
        input_dim: int,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        dropout: float = 0.3,
        cross_layer_num: int = 2,
    ):
        super().__init__()

        # Build pyramid MLP: input_dim -> ... -> last_hidden_units
        hidden_dims = self._compute_hidden_dims(input_dim, last_hidden_units, num_hidden_layers)
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.cross_net = CrossNet(input_dim=in_dim, layer_num=cross_layer_num)

        # Output: concat MLP output + CrossNet output -> 1
        self.output_layer = nn.Linear(in_dim + in_dim, 1)

    def forward(self, x):
        mlp_out = self.mlp(x)
        cross_out = self.cross_net(mlp_out)
        combined = torch.cat([mlp_out, cross_out], dim=-1)
        return self.output_layer(combined).squeeze(-1)

    @staticmethod
    def _compute_hidden_dims(input_dim, last_hidden_units, num_hidden_layers):
        """Compute pyramid hidden dimensions, halving from input_dim down to last_hidden_units."""
        if num_hidden_layers <= 0:
            return []
        if num_hidden_layers == 1:
            return [last_hidden_units]
        dims = []
        for i in range(num_hidden_layers):
            ratio = (i + 1) / num_hidden_layers
            dim = int(input_dim + (last_hidden_units - input_dim) * ratio)
            dims.append(max(dim, last_hidden_units))
        return dims


class RankerNet(nn.Module):
    """Multi-task ranker network with DCN towers.

    Three task towers: CTR, Click-to-Action, Action-to-Order.
    Derived probabilities: CTCAR, CTCVR, CVR.
    """

    def __init__(
        self,
        user_input_size: int,
        item_input_size: int,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        dropout: float = 0.3,
        cross_layer_num: int = 2,
    ):
        super().__init__()
        self.user_input_size = user_input_size
        self.item_input_size = item_input_size

        combined_dim = user_input_size + item_input_size

        self.tower_ctr = Tower(
            input_dim=combined_dim,
            num_hidden_layers=num_hidden_layers,
            last_hidden_units=last_hidden_units,
            dropout=dropout,
            cross_layer_num=cross_layer_num,
        )
        self.tower_click_to_action = Tower(
            input_dim=combined_dim,
            num_hidden_layers=num_hidden_layers,
            last_hidden_units=last_hidden_units,
            dropout=dropout,
            cross_layer_num=cross_layer_num,
        )
        self.tower_action_to_order = Tower(
            input_dim=combined_dim,
            num_hidden_layers=num_hidden_layers,
            last_hidden_units=last_hidden_units,
            dropout=dropout,
            cross_layer_num=cross_layer_num,
        )

    def forward(self, user_features, item_features):
        """
        Args:
            user_features: (batch, user_input_size) aggregated user representation.
            item_features: (batch, item_input_size) item feature vector.

        Returns:
            logit_ctr, logit_click_to_action, logit_action_to_order: each (batch,)
        """
        x = torch.cat([user_features, item_features], dim=-1)

        logit_ctr = self.tower_ctr(x)
        logit_click_to_action = self.tower_click_to_action(x)
        logit_action_to_order = self.tower_action_to_order(x)

        return logit_ctr, logit_click_to_action, logit_action_to_order

    def set_dropout(self, dropout: float):
        """Update dropout rate for all towers (used for post-epoch adjustment)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    def freeze_mlp(self):
        """Freeze MLP parameters in all towers (keep CrossNet and output trainable)."""
        for tower in [self.tower_ctr, self.tower_click_to_action, self.tower_action_to_order]:
            for param in tower.mlp.parameters():
                param.requires_grad = False
