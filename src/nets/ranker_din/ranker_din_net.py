import torch
import torch.nn as nn

from .din import AttentionSequencePoolingLayer


class CrossNet(nn.Module):
    """Cross Network from DCN-V2.

    Learns explicit feature interactions via cross layers:
        x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l

    Reference: https://arxiv.org/abs/2008.13535
    """

    def __init__(self, in_features: int, layer_num: int = 2):
        super().__init__()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(in_features, in_features) for _ in range(layer_num)]
        )
        for layer in self.linear_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, D)

        Returns:
            x_l: (B, D) cross-network output.
        """
        x_l = inputs
        for layer in self.linear_layers:
            xl_w = layer(x_l)
            x_l = inputs * xl_w + x_l
        return x_l


def _make_hidden_layers(
    input_size: int, num_layers: int, last_hidden_units: int
) -> nn.Sequential:
    """Build MLP hidden layers with BatchNorm and ELU activation.

    Layer sizes decrease by powers of 2 from input to last_hidden_units.
    """
    modules = []
    for i in range(num_layers):
        if i == 0:
            in_dim = input_size
        else:
            in_dim = last_hidden_units * 2 ** (num_layers - i)
        out_dim = last_hidden_units * 2 ** (num_layers - 1 - i)

        modules.append(nn.Linear(in_dim, out_dim, bias=False))
        modules.append(nn.BatchNorm1d(out_dim))
        modules.append(nn.ELU())
    return nn.Sequential(*modules)


class RankerDinNet(nn.Module):
    """Multi-task ranker network with DIN (Deep Interest Network) attention.

    Architecture per task tower:
        - Without stacked DCN (default):
            concat(hidden_output, cross_output) -> output_layer
        - With stacked DCN:
            cross -> hidden -> output_layer

    DIN attention is used to pool user click history into a fixed-size
    representation conditioned on the candidate item.

    The network outputs three logits for the multi-task decomposition:
        CTR (impression -> click),
        click_to_action (click -> action),
        action_to_order (action -> purchase).
    """

    def __init__(
        self,
        user_input_size: int,
        item_input_size: int,
        click_embed_size: int,
        num_hidden_layers: int = 1,
        last_hidden_units: int = 256,
        dropout: float = 0.3,
        cross_layer_num: int = 2,
        din_projection_dim: int = 64,
        use_stacked_dcn: bool = False,
    ):
        """
        Args:
            user_input_size: Dimension of non-click user features (e.g. search query).
            item_input_size: Dimension of item features.
            click_embed_size: Dimension of concatenated click history embeddings
                (item_embed + text_embed + market_embed + category_embed).
            num_hidden_layers: Number of MLP hidden layers per tower.
            last_hidden_units: Size of the final hidden layer.
            dropout: Dropout probability for the input.
            cross_layer_num: Number of cross layers in CrossNet.
            din_projection_dim: Projection dimension for DIN attention.
                Set to 0 to disable projection.
            use_stacked_dcn: If True, use stacked DCN (cross -> deep -> output).
                If False, use parallel DCN (concat(deep, cross) -> output).
        """
        super().__init__()
        self.use_stacked_dcn = use_stacked_dcn

        # DIN attention pooling for click history
        self.din_pool = AttentionSequencePoolingLayer(
            att_hidden_units=36,
            weight_normalization=True,
            embedding_dim=click_embed_size,
            projection_dim=din_projection_dim,
        )

        # Total DNN input = user_features + din_pooled_click + item_features
        dnn_input_size = user_input_size + click_embed_size + item_input_size

        self.dropout = nn.Dropout(dropout)

        # Multi-task towers
        task_names = ["ctr", "click_to_action", "action_to_order"]
        self.tower_hidden_layers = nn.ModuleDict(
            {
                name: _make_hidden_layers(dnn_input_size, num_hidden_layers, last_hidden_units)
                for name in task_names
            }
        )
        self.tower_cross_layers = nn.ModuleDict(
            {
                name: CrossNet(in_features=dnn_input_size, layer_num=cross_layer_num)
                for name in task_names
            }
        )

        if use_stacked_dcn:
            output_layer_input_dim = last_hidden_units
        else:
            output_layer_input_dim = last_hidden_units + dnn_input_size

        self.out_layers = nn.ModuleDict(
            {name: nn.Linear(output_layer_input_dim, 1) for name in task_names}
        )

    def _run_tower(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """Run a single task tower.

        Args:
            x: (B, D) DNN input.
            task_name: One of "ctr", "click_to_action", "action_to_order".

        Returns:
            logit: (B, 1) raw logit for the task.
        """
        if self.use_stacked_dcn:
            x = self.tower_cross_layers[task_name](x)
            dnn_out = self.tower_hidden_layers[task_name](x)
            logit = self.out_layers[task_name](dnn_out)
        else:
            dnn_out = self.tower_hidden_layers[task_name](x)
            cross_out = self.tower_cross_layers[task_name](x)
            logit = self.out_layers[task_name](torch.cat([dnn_out, cross_out], dim=1))
        return logit

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        click_query: torch.Tensor,
        click_keys: torch.Tensor,
        click_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            user_features: (B, user_input_size) non-click user features.
            item_features: (B, item_input_size) item features.
            click_query: (B, 1, click_embed_size) candidate item embedding for DIN query.
            click_keys: (B, T, click_embed_size) click history embeddings for DIN keys.
            click_mask: (B, T) boolean mask for valid click positions.

        Returns:
            logit_ctr: (B, 1) CTR logit.
            logit_c2a: (B, 1) click-to-action logit.
            logit_a2o: (B, 1) action-to-order logit.
        """
        # DIN attention pooling: (B, 1, click_embed_size) -> (B, click_embed_size)
        pooled_click = self.din_pool(click_query, click_keys, click_mask)
        pooled_click = pooled_click.squeeze(1)  # (B, click_embed_size)

        # Concatenate all inputs
        dnn_input = torch.cat([user_features, pooled_click, item_features], dim=-1)
        x = self.dropout(dnn_input)

        logit_ctr = self._run_tower(x, "ctr")
        logit_c2a = self._run_tower(x, "click_to_action")
        logit_a2o = self._run_tower(x, "action_to_order")
        return logit_ctr, logit_c2a, logit_a2o

    def set_dropout(self, p: float) -> None:
        """Update the input dropout probability."""
        self.dropout = nn.Dropout(p)
