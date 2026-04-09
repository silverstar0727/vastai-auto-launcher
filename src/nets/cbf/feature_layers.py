from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from utils.constants import UNKNOWN_ITEM_INDEX
from utils.seed import get_num_valid_tokens


class FeatureLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.serving = False

    def predict(self, features: Dict[str, torch.tensor]) -> Optional[torch.tensor]:
        input_tensors = []
        for feat in self.feature_names:
            feat_tensors = features.get(feat, None)
            if feat_tensors is not None:
                input_tensors.append(feat_tensors)
        if not input_tensors:
            return None

        if self.serving:
            feat_tensors = []
            for t in input_tensors:
                device = t.device
                feat_tensors.append(self.forward(t.to("cpu")).to(device))
        else:
            feat_tensors = [self.forward(t) for t in input_tensors]

        feat_tensors = torch.stack(feat_tensors, dim=1)
        return feat_tensors

    def forward(self, x):
        raise NotImplementedError

    def set_serving_mode(self):
        self.serving = True


class Polynomial(FeatureLayer):
    def __init__(self, feature_names, degrees=[0.5, 1, 2]):
        super().__init__()
        self.feature_names = feature_names
        self.degrees = degrees

    def forward(self, x):
        xs = [x**d for d in self.degrees]
        xs = torch.cat(xs, dim=-1)
        return xs


class MeanEmbedding(FeatureLayer):
    def __init__(
        self,
        feature_names,
        num_embeddings,
        embedding_dim,
        embeddings: Optional[np.array] = None,
        fc_head: int = 0,
        padding_idx=UNKNOWN_ITEM_INDEX,
    ):
        super().__init__()
        if embeddings is None:
            self.embeddings = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        else:
            self.embeddings = torch.nn.Embedding.from_pretrained(
                torch.FloatTensor(embeddings),
                freeze=True,
            )
        hidden_modules = []
        if fc_head > 0:
            hidden_modules.append(nn.Linear(embedding_dim, fc_head, bias=False))
            hidden_modules.append(nn.BatchNorm1d(fc_head))
            hidden_modules.append(nn.ELU())
        self.hidden_layers = nn.Sequential(*hidden_modules)
        self.feature_names = feature_names

    def forward(self, x):
        x = torch.where(x >= self.embeddings.num_embeddings, torch.zeros_like(x), x)

        if self.serving:
            input_tensor = torch.nn.functional.embedding_bag(x, self.embeddings.weight)
        else:
            num_items = get_num_valid_tokens(x)
            num_items = torch.max(num_items, torch.ones_like(num_items))
            embeddings = self.embeddings(x)
            embeddings = torch.sum(embeddings, 1)
            input_tensor = embeddings / num_items
        return self.hidden_layers(input_tensor)

    def get_embed_tensor(self) -> torch.Tensor:
        return self.embeddings.weight


class FeatureInputLayerName(object):
    ITEM_EMBEDDING = "item_embed_layer"
    TEXT_EMBEDDING = "text_embed_layer"
    USER_AGE = "user_age_layer"
    MARKET_EMBEDDING = "market_embed_layer"
    CATEGORY_EMBEDDING = "category_embed_layer"
    PRICE_GROUP_EMBEDDING = "price_group_embed_layer"
    ITEM_TAG_VECTOR = "item_tag_vector"
    ATTR_VECTOR = "attr_vector"
    TOKEN_EMBEDDING = "token_embed_layer"
    ITEM_CTR_SCORE = "item_ctr_score_layer"
    ITEM_CLICK_SCORE = "item_click_score_layer"
    ITEM_REVIEW_SCORE = "item_review_score_layer"
    ITEM_POS_REVIEW_SCORE = "item_pos_review_score_layer"
    ITEM_PRICE_SCORE = "item_price_score_layer"
    ITEM_DISPLAY_SCORE = "item_display_score_layer"


class Feature:
    def __init__(self):
        raise NotImplementedError

    def get_layer_name(self):
        return self.layer_name

    def get_feature_name(self):
        return self.name

    def get_hyperparameters(self):
        raise NotImplementedError

    def get_output_size(self):
        raise NotImplementedError

    @classmethod
    def from_hparams(cls, hparams: Dict[str, Any]):
        cls_name = hparams.pop("cls")
        if cls_name == SparseFeat.cls_name:
            return SparseFeat(**hparams)
        elif cls_name == PolynomialFeat.cls_name:
            return PolynomialFeat(**hparams)
        else:
            raise ValueError


class SparseFeat(Feature):
    cls_name = "SparseFeat"

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        embedding_dim: int,
        embeddings_initializer: Optional[np.array] = None,
        fc_head: int = 0,
        layer_name: Optional[str] = None,
        padding_idx: Optional[int] = None,
    ):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embedding_initializer = embeddings_initializer
        if layer_name is None:
            self.layer_name = name
        else:
            self.layer_name = layer_name
        self.padding_idx = padding_idx
        self.fc_head = fc_head

    def get_hyperparameters(self):
        hparams = {
            "cls": self.cls_name,
            "name": self.name,
            "layer_name": self.layer_name,
            "embedding_dim": self.embedding_dim,
            "vocabulary_size": self.vocabulary_size,
        }
        return hparams

    def get_output_size(self):
        if self.fc_head > 0:
            return self.fc_head
        else:
            return self.embedding_dim


class PolynomialFeat(Feature):
    cls_name = "PolynomialFeat"

    def __init__(self, name: str, layer_name: str, degrees: List[float] = [0.5, 1, 2]):
        self.name = name
        self.layer_name = layer_name
        self.degrees = degrees

    def get_hyperparameters(self):
        hparams = {
            "cls": self.cls_name,
            "name": self.name,
            "layer_name": self.layer_name,
            "degrees": self.degrees,
        }
        return hparams

    def get_output_size(self):
        return len(self.degrees)


def build_feature_layers(features: List[Feature]):
    feature_layers = OrderedDict()

    for feat in features:
        layer_name = feat.get_layer_name()
        feature_name = feat.get_feature_name()

        if layer_name in feature_layers.keys():
            feature_layers[layer_name].feature_names.append(feature_name)

        else:
            if isinstance(feat, SparseFeat):
                feature_layers[layer_name] = MeanEmbedding(
                    num_embeddings=feat.vocabulary_size,
                    embedding_dim=feat.embedding_dim,
                    feature_names=[feature_name],
                    embeddings=feat.embedding_initializer,
                    fc_head=feat.fc_head,
                    padding_idx=feat.padding_idx,
                )

            elif isinstance(feat, PolynomialFeat):
                feature_layers[layer_name] = Polynomial(feature_names=[feature_name], degrees=feat.degrees)
            else:
                raise ValueError
    return feature_layers


def get_process_output_size(features: List[Feature], feat_names: Optional[List[str]]):
    output_size = 0
    for feat in features:
        if feat_names is None:
            output_size += feat.get_output_size()
        else:
            if feat.name in feat_names:
                output_size += feat.get_output_size()
    return output_size


class Checkpoint:
    MODEL_HYPER_PARAMETER_KEY = "hyper_parameters"
    FEATURE_HYPER_PARAMETER_KEY = "feat_hparams"

    @classmethod
    def load_hparams(cls, ckpt_filename, map_location=None):
        state_dict = torch.load(ckpt_filename, map_location)
        hparams = state_dict[cls.MODEL_HYPER_PARAMETER_KEY]
        feat_hparams = hparams.pop(cls.FEATURE_HYPER_PARAMETER_KEY)

        features = [Feature.from_hparams(hparam) for hparam in feat_hparams]
        hparams["features"] = features
        return hparams

    @classmethod
    def get_hparams(cls, features, **kwargs):
        hparams = {
            cls.FEATURE_HYPER_PARAMETER_KEY: [feat.get_hyperparameters() for feat in features],
        }
        hparams.update(kwargs)
        return hparams
