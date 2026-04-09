from typing import Dict, List

import numpy as np
import pandas as pd
from utils.constants import UNKNOWN_ITEM_INDEX, DatasetField, FeatureField


def _encode_texts(texts, pretrained_text_path=None):
    """SentenceTransformer를 사용한 텍스트 임베딩 생성."""
    from sentence_transformers import SentenceTransformer
    from utils.constants import TEXT_MODEL_NAME

    model_name = pretrained_text_path if pretrained_text_path else TEXT_MODEL_NAME
    model = SentenceTransformer(model_name)
    all_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return np.array(all_embeddings)


class TokenEmbedProcessor:
    """토큰 단위 임베딩의 mean pooling을 수행하는 프로세서."""

    def __init__(self, pretrained_text_path):
        import json
        import os
        import sentencepiece as spm
        from utils.constants import TOKENIZER_FILE

        self.pretrained_text_path = pretrained_text_path

        # Load tokenizer
        tokenizer_path = os.path.join(pretrained_text_path, TOKENIZER_FILE)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

        # Load token index
        token_index_path = os.path.join(pretrained_text_path, "token_index.json")
        with open(token_index_path, "r") as f:
            self.token_index = json.load(f)

        # Load token embeddings
        token_emb_path = os.path.join(pretrained_text_path, "token_emb.npy")
        self.token_emb = np.load(token_emb_path)

    def run(self, text, reduce_mean=True):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            text = ""
        tokens = self.sp.EncodeAsPieces(str(text))
        indices = []
        for token in tokens:
            idx = self.token_index.get(token, None)
            if idx is not None:
                indices.append(idx)

        if not indices:
            embed_dim = self.token_emb.shape[1]
            return np.zeros(embed_dim)

        embeddings = self.token_emb[indices]
        if reduce_mean:
            return embeddings.mean(axis=0)
        return embeddings


class GoodsInfo(object):
    def __init__(self, df_items):
        self.df = df_items

    def get_text_embeddings_arr(self, pretrained_text_path=None):
        df = self.df.sort_values(DatasetField.ITEM_INDEX)
        texts = list(df.loc[:, DatasetField.ITEM_TEXT])

        all_embeddings = _encode_texts(texts, pretrained_text_path)
        embed_dim = all_embeddings.shape[-1]
        all_embeddings = np.concatenate([np.zeros((1, embed_dim)), all_embeddings])
        return all_embeddings

    def get_text_embeddings(self, pretrained_text_path=None):
        item_indexes = list(self.df[DatasetField.ITEM_INDEX])
        texts = list(self.df.loc[:, DatasetField.ITEM_TEXT])
        all_embeddings = _encode_texts(texts, pretrained_text_path)
        return dict(zip(item_indexes, all_embeddings))

    def get_valid_num_items(self):
        return len(self.df)

    def get_item_side_features(self, insert_mask_token=False) -> Dict[str, np.array]:
        df_items = self.df.set_index(DatasetField.ITEM_INDEX)
        if UNKNOWN_ITEM_INDEX not in df_items.index:
            df_items.loc[UNKNOWN_ITEM_INDEX] = [0] * len(df_items.columns)
        df_items = df_items.sort_index()

        markets = df_items[DatasetField.MARKET_INDEX].tolist()
        categories = df_items[DatasetField.CATEGORY_INDEX].tolist()
        standard_categories = df_items[DatasetField.STANDARD_CATEGORY_INDEX].tolist()
        market_types = df_items[DatasetField.MARKET_TYPE_INDEX].tolist()
        if insert_mask_token:
            markets.append(0)
            categories.append(0)
            standard_categories.append(0)
            market_types.append(0)

        item_side_features = {
            FeatureField.ITEM_MARKET: np.array(markets).reshape(-1, 1),
            FeatureField.ITEM_CATEGORY: np.array(categories).reshape(-1, 1),
            FeatureField.ITEM_STANDARD_CATEGORY: np.array(standard_categories).reshape(-1, 1),
            FeatureField.ITEM_MARKET_TYPE: np.array(market_types).reshape(-1, 1),
        }
        return item_side_features

    def get_item_side_features_selectively(
        self,
        features: List,
        insert_mask_token: bool = False,
    ) -> Dict:
        df_items = self.df.set_index(DatasetField.ITEM_INDEX)
        if UNKNOWN_ITEM_INDEX not in df_items.index:
            df_items.loc[UNKNOWN_ITEM_INDEX] = [0] * len(df_items.columns)
        df_items = df_items.sort_index()

        feature_to_field = {
            FeatureField.ITEM_MARKET: DatasetField.MARKET_INDEX,
            FeatureField.ITEM_CATEGORY: DatasetField.CATEGORY_INDEX,
            FeatureField.ITEM_STANDARD_CATEGORY: DatasetField.STANDARD_CATEGORY_INDEX,
            FeatureField.ITEM_PRICE: DatasetField.ITEM_PRICE,
        }

        item_side_features = {}
        for feature in features:
            field = feature_to_field.get(feature, None)
            if field is None:
                continue
            values = df_items[field].tolist()
            if insert_mask_token:
                values.append(0)
            item_side_features[feature] = np.array(values).reshape(-1, 1)
        return item_side_features

    def get_token_embeddings_arr(
        self,
        pretrained_text_path: str,
        append_padding_index: bool = True,
        num_special_index: int = 0,
        do_normalize: bool = True,
        pretrained_emb: str = "inhouse",
    ) -> np.array:
        df = self.df.sort_values(DatasetField.ITEM_INDEX)
        texts = list(df.loc[:, DatasetField.ITEM_TEXT])

        if pretrained_emb == 'inhouse':
            processor = TokenEmbedProcessor(pretrained_text_path)
            all_embeddings = np.array([processor.run(text) for text in texts])
        elif pretrained_emb == 'sbert':
            all_embeddings = _encode_texts([text if not pd.isna(text) else "" for text in texts], pretrained_text_path)
        else:
            raise Exception("invalid pretrained embedding type")

        if do_normalize:
            all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1).reshape(-1, 1)

        embed_dim = all_embeddings.shape[-1]
        if append_padding_index:
            all_embeddings = np.concatenate([np.zeros((1, embed_dim)), all_embeddings])

        if num_special_index > 0:
            all_embeddings = np.concatenate([all_embeddings, np.zeros((num_special_index, embed_dim))])
        return all_embeddings

    def get_item_categories(self, append_padding_index: bool = True) -> np.array:
        df_items = self.df.sort_values(DatasetField.ITEM_INDEX)
        item_categories = df_items[DatasetField.CATEGORY_INDEX].values.tolist()
        if append_padding_index:
            item_categories = [0] + item_categories
        item_categories = np.array(item_categories)
        return item_categories

    def get_item_standard_categories(self, append_padding_index: bool = True) -> np.array:
        df_items = self.df.sort_values(DatasetField.ITEM_INDEX)
        item_standard_categories = df_items[DatasetField.STANDARD_CATEGORY_INDEX].values.tolist()
        if append_padding_index:
            item_standard_categories = [0] + item_standard_categories
        item_standard_categories = np.array(item_standard_categories)
        return item_standard_categories
