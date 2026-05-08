"""이미지 태그 속성(attribute) 벡터 인코딩 유틸리티.

기존 reco_common.util.ml.feature.attribute 에서 포팅.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import DatasetField, FeatureField
from utils.df_utils import read_table
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix
from nets.cbf.feature_layers import FeatureInputLayerName, SparseFeat

ITEM_ID = "goods_sno"
ATTR_FIELD_ID = "goods_attribute_field_value_sno"
ATTR_FIELD_INDEX = "attr_index"
ATTR_FIELD_CONFIDENCE = "predict_confidence"


def encode_attribute(item_ids, attribute_meta_file, item2attr_file, insert_unk_pad=False):
    attr_processor = AttrProcessor(attribute_meta_file)
    attr_vectors = attr_processor.get_vector(item_ids, item2attr_file)
    if insert_unk_pad:
        attr_vectors = np.concatenate(
            [np.zeros((1, attr_vectors.shape[-1]), dtype=np.float32), attr_vectors],
            axis=0,
        )
    return attr_vectors


class AttrProcessor:
    def __init__(self, attribute_meta_file):
        df_attr_meta = read_table(attribute_meta_file)
        attr_index2sno = df_attr_meta["sno"].to_dict()
        self.attr_sno2index = {v: k for k, v in attr_index2sno.items()}

    def get_vector(self, item_ids, item2attr_filename):
        df_goods2attr = read_table(item2attr_filename)
        df_goods2attr = df_goods2attr[df_goods2attr[ITEM_ID].isin(item_ids)]
        df_goods2attr[ATTR_FIELD_INDEX] = df_goods2attr[ATTR_FIELD_ID].map(self.attr_sno2index)
        df_goods2attr = df_goods2attr.loc[df_goods2attr[ATTR_FIELD_INDEX].notnull()]
        df_goods2attr[ATTR_FIELD_INDEX] = df_goods2attr[ATTR_FIELD_INDEX].astype(int)
        df_goods2attr[ATTR_FIELD_CONFIDENCE] = df_goods2attr[ATTR_FIELD_CONFIDENCE].fillna(1.0)
        df_goods2attr = df_goods2attr.groupby(ITEM_ID).agg(lambda x: list(x))
        attr_vectors = np.zeros((len(item_ids), len(self.attr_sno2index)), dtype=np.float32)
        for i, item in tqdm(enumerate(item_ids)):
            df = df_goods2attr.loc[df_goods2attr.index == item]
            attr_vector = np.zeros_like(attr_vectors[0])
            if len(df) > 0:
                df = df.iloc[0]
                attr_vector[np.array(df[ATTR_FIELD_INDEX])] = np.array(df[ATTR_FIELD_CONFIDENCE])
            attr_vectors[i] = attr_vector
        return attr_vectors


def build_attr_feat(model_path, df_items, attr_meta_file, attr_file):
    """이미지 태그를 사용하는 Feature 객체를 만들어 주는 함수."""
    label_encoder = LabelEncoder.from_model_dir(model_path, LabelEncoderPrefix.ITEM)
    item_ids = label_encoder.to_ids(df_items[DatasetField.ITEM_INDEX])
    attr_vectors = encode_attribute(item_ids, attr_meta_file, attr_file, insert_unk_pad=True)
    vocabulary_size, embedding_dim = attr_vectors.shape

    attr_feat = SparseFeat(
        FeatureField.ITEM,
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        layer_name=FeatureInputLayerName.ATTR_VECTOR,
        embeddings_initializer=attr_vectors,
    )
    user_tower_attr_feat = SparseFeat(
        FeatureField.CLICK_ITEMS,
        vocabulary_size=embedding_dim,
        embedding_dim=embedding_dim,
        layer_name=FeatureInputLayerName.ATTR_VECTOR,
        embeddings_initializer=attr_vectors,
    )
    return attr_feat, user_tower_attr_feat
