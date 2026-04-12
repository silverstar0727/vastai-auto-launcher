"""Complement 모델의 Two-Tower 네트워크.

CBFNet과 동일한 구조이지만 _get_click_items_side_feat에서
ITEM_STANDARD_CATEGORY 대신 ITEM_CATEGORY를 사용한다.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from nets.cbf.feature_layers import (
    Checkpoint,
    Feature,
    build_feature_layers,
    get_process_output_size,
)
from nets.cbf.two_tower import create_tower_layers
from utils.constants import CKPT_FILENAME, FeatureField


def _get_click_items_side_feat(
    click_items: torch.Tensor,
    user_feat_names: List[str],
    item_feat_values: Dict[str, np.array],
    device,
) -> Dict[str, torch.Tensor]:
    """user 타워에 입력하는 item feature를 가져온다.

    CBFNet의 _get_click_items_side_feat와 동일하지만
    카테고리 lookup에 ITEM_CATEGORY를 사용한다 (ITEM_STANDARD_CATEGORY 대신).
    """
    item_feats = {}

    if FeatureField.CLICK_MARKETS in user_feat_names:
        item2market = torch.from_numpy(
            item_feat_values.get(FeatureField.ITEM_MARKET).reshape(-1,)
        ).to(device)
        item_feats[FeatureField.CLICK_MARKETS] = item2market[click_items]

    if FeatureField.CLICK_CATEGORIES in user_feat_names:
        item2category = torch.from_numpy(
            item_feat_values.get(FeatureField.ITEM_CATEGORY).reshape(-1,)
        ).to(device)
        item_feats[FeatureField.CLICK_CATEGORIES] = item2category[click_items]

    return item_feats


class ComplementNet(nn.Module):
    """Complement Two-Tower 네트워크.

    CBFNet과 동일한 아키텍처이지만:
    - 카테고리 side feat에 ITEM_CATEGORY 사용 (ITEM_STANDARD_CATEGORY 대신)
    """

    def __init__(
        self,
        num_items: int,
        features: List[Feature],
        user_tower_features: List[str],
        item_tower_features: List[str],
        item_feat_values: Dict[str, np.array],
        num_hidden_layers=1,
        last_hidden_units=256,
        item_dropout_prob=0,
        normalize_outputs=False,
    ):
        super().__init__()
        self.num_items = num_items
        self.user_tower_features = user_tower_features
        self.item_tower_features = item_tower_features
        self.item_feat_values = item_feat_values
        self.item_dropout_prob = item_dropout_prob
        self.normalize_outputs = normalize_outputs
        self.serving = False

        self._hparams = Checkpoint.get_hparams(
            features,
            user_tower_features=user_tower_features,
            item_tower_features=item_tower_features,
            num_items=num_items,
            num_hidden_layers=num_hidden_layers,
            last_hidden_units=last_hidden_units,
            normalize_outputs=normalize_outputs,
        )

        self.feature_process_modules = torch.nn.ModuleDict(build_feature_layers(features))
        self.user_tower = create_tower_layers(
            features, user_tower_features, num_hidden_layers, last_hidden_units,
            l2_norm=normalize_outputs,
        )
        self.item_tower = create_tower_layers(
            features, item_tower_features, num_hidden_layers, last_hidden_units,
            l2_norm=normalize_outputs,
        )

    def forward(self, user_features: Dict, item_idxes: Optional[torch.Tensor] = None):
        side_feats = _get_click_items_side_feat(
            click_items=user_features[FeatureField.CLICK_ITEMS],
            user_feat_names=self.get_input_feature_names(),
            item_feat_values=self.item_feat_values,
            device=next(self.parameters()).device if list(self.parameters()) else torch.device("cpu"),
        )
        user_features.update(side_feats)

        user_inputs = self.get_dnn_inputs(user_features)
        user_emb = self.user_tower(user_inputs)
        if self.serving:
            return user_emb

        item_emb = self.get_item_tower_embed(item_idxes)
        x = torch.matmul(user_emb, item_emb.T)
        return x

    def get_item_tower_embed(self, item_idxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        if item_idxes is not None:
            target_item_idxes = item_idxes.reshape(-1,)
        else:
            target_item_idxes = torch.arange(0, self.num_items, dtype=torch.int64, device=device)

        item_feat_tensors = {
            FeatureField.ITEM: target_item_idxes.reshape(-1, 1),
        }

        if self.item_dropout_prob > 0 and self.training:
            preserve_prob = 1.0 - self.item_dropout_prob
            items_dropped = torch.empty_like(target_item_idxes).bernoulli_(preserve_prob) * target_item_idxes
        else:
            items_dropped = target_item_idxes
        item_feat_tensors[FeatureField.ITEM_DROPOUT] = items_dropped.reshape(-1, 1)

        for feat_name, feat_value in self.item_feat_values.items():
            target_feat_values = feat_value[target_item_idxes.cpu().numpy()]
            item_feat_tensors[feat_name] = torch.from_numpy(target_feat_values).to(device)

        item_tower_inputs = self.get_dnn_inputs(item_feat_tensors)
        item_emb = self.item_tower(item_tower_inputs)
        return item_emb

    def get_input_feature_names(self) -> List[str]:
        input_feat_names = []
        for feat_layer in self.feature_process_modules.values():
            input_feat_names += feat_layer.feature_names
        input_feat_names = list(set(input_feat_names))
        return input_feat_names

    def get_dnn_inputs(self, feat_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        dnn_inputs = []
        for input_layer in self.feature_process_modules.values():
            feat_tensor = input_layer.predict(feat_tensors)
            if feat_tensor is not None:
                feat_tensor = feat_tensor.reshape(feat_tensor.shape[0], -1)
                dnn_inputs.append(feat_tensor)
        dnn_inputs = torch.cat(dnn_inputs, dim=-1)
        return dnn_inputs

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination, prefix, keep_vars)
        destination.update({"item_feat_values": self.item_feat_values})
        return destination

    def save_checkpoint(self, ckpt_path):
        ckpt = {"hyper_parameters": self._hparams, "state_dict": self.state_dict()}
        torch.save(ckpt, ckpt_path)

    @classmethod
    def load_from_ckpt(cls, model_dir, map_location="cpu"):
        checkpoint_path = os.path.join(model_dir, CKPT_FILENAME)
        state_dict = torch.load(checkpoint_path, map_location)
        hparams = Checkpoint.load_hparams(checkpoint_path, map_location)
        hparams["item_feat_values"] = state_dict["state_dict"]["item_feat_values"]
        model = cls(**hparams)
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        return model
