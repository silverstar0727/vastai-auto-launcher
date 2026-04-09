import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import lightning as L

from utils.constants import DatasetField, FeatureField, RawDataField
from utils.goods_info import TokenEmbedProcessor, _encode_texts
from utils.label_encoder import LabelEncoder, LabelEncoderPrefix
from utils.preprocess import default_load_raw_df_items
from utils.price import add_item_price_group_column

logger = logging.getLogger(__name__)

ITEM_TOWER_OUTPUT_FILENAME = "item_embed"


class SaveItemEmbedCallback(L.Callback):
    """학습 완료 후 아이템 타워 임베딩을 저장하는 Callback."""

    def __init__(
        self,
        model_path: str = "",
        category_file: str = "",
        standard_category_file: str = "",
        goods_file: str = "",
        use_item_emb_in_item_tower: bool = False,
        inhouse_text_path: Optional[str] = None,
        expand_oov: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.category_file = category_file
        self.standard_category_file = standard_category_file
        self.goods_file = goods_file
        self.use_item_emb_in_item_tower = use_item_emb_in_item_tower
        self.inhouse_text_path = inhouse_text_path
        self.expand_oov = expand_oov

    def on_fit_end(self, trainer, pl_module):
        if not self.model_path or not self.goods_file:
            logger.warning("SaveItemEmbedCallback: model_path or goods_file not set, skipping.")
            return

        save_item_tower_embed(
            self.category_file,
            self.standard_category_file,
            self.goods_file,
            self.model_path,
            self.use_item_emb_in_item_tower,
            inhouse_text_path=self.inhouse_text_path,
            expand_oov=self.expand_oov,
        )


def save_item_tower_embed(
    category_file,
    standard_category_file,
    goods_file,
    model_path,
    use_item_emb_in_item_tower: bool,
    inhouse_text_path=None,
    attr_file_path: Optional[Dict] = None,
    expand_oov: bool = True,
    max_batch_size: int = 100000,
):
    from nets.cbf.two_tower import CBFNet

    cbf = CBFNet.load_from_ckpt(model_path)
    cbf.eval()

    feature_names = cbf.get_input_feature_names()

    oov_df_items = get_cold_start_items(goods_file, category_file, standard_category_file, model_path, feature_names)
    logger.info(f"number of OOV items: {len(oov_df_items)}")
    if len(oov_df_items) == 0 or not expand_oov:
        logger.info("no out-of-vocabulary items")
        _save_npy_file(model_path, cbf.get_item_tower_embed())
        return

    on_vocab_items = LabelEncoder.from_model_dir(model_path, LabelEncoderPrefix.ITEM).get_valid_item_ids()
    output_item_encoder = LabelEncoder.from_item_ids(
        pd.Series(list(on_vocab_items) + list(oov_df_items[RawDataField.ITEM_ID]))
    )
    output_item_encoder.save(model_path, LabelEncoderPrefix.OUTPUT_ITEM)

    with torch.inference_mode():
        output_list = []
        for i in range(0, len(oov_df_items), max_batch_size):
            oov_df_items_ = oov_df_items.iloc[i : i + max_batch_size]
            oov_item_tower_output = _run_on_single_batch(
                cbf, oov_df_items_, inhouse_text_path, attr_file_path, use_item_emb_in_item_tower, feature_names,
            )
            output_list.append(oov_item_tower_output)

        oov_item_tower_embed = torch.cat(output_list, dim=0)
        item_tower_embed = cbf.get_item_tower_embed()
        full_item_tower_embed = torch.cat([item_tower_embed, oov_item_tower_embed], dim=0)

    _save_npy_file(model_path, full_item_tower_embed)
    logger.info("SAVE is done")


def get_cold_start_items(goods_filename, category_filename, standard_category_filename, saved_model_path, feature_names):
    item_encoder = LabelEncoder.from_model_dir(saved_model_path, LabelEncoderPrefix.ITEM)
    on_vocab_items = item_encoder.get_valid_item_ids()
    oov_df_items = _get_new_items(
        goods_filename, category_filename, standard_category_filename, saved_model_path, on_vocab_items
    )
    oov_df_items = _preprocess_item_feature(oov_df_items, saved_model_path, feature_names)
    return oov_df_items


def _get_new_items(goods_filename, category_filename, standard_category_filename, model_path, cur_existing_items):
    df_items = default_load_raw_df_items(
        goods_filename=goods_filename,
        category_filename=category_filename,
        standard_category_filename=standard_category_filename,
        append_category_text=False,
        append_standard_category_text=True,
    )
    df_new_items = df_items[~df_items[RawDataField.ITEM_ID].isin(cur_existing_items)]
    logger.info(f'df_new_items.shape: {df_new_items.shape}')
    return df_new_items


def _preprocess_item_feature(df_items, model_path, feature_names):
    market_encoder = LabelEncoder.from_model_dir(model_path, LabelEncoderPrefix.MARKET)
    category_encoder = LabelEncoder.from_model_dir(model_path, LabelEncoderPrefix.STANDARD_CATEGORY)

    df_items[FeatureField.ITEM_MARKET] = (
        df_items[RawDataField.MARKET_ID].astype(str).map(market_encoder.item_id2index).fillna(0).astype(int)
    )
    df_items[FeatureField.ITEM_STANDARD_CATEGORY] = (
        df_items[RawDataField.STANDARD_CATEGORY_ID].astype(str).map(category_encoder.item_id2index).fillna(0).astype(int)
    )

    if FeatureField.ITEM_PRICE_GROUP in feature_names:
        df_items[FeatureField.ITEM_PRICE] = df_items[RawDataField.ITEM_PRICE]
        df_category_price_percentile = pd.read_pickle(
            os.path.join(model_path, 'standard_category_price_percentile.pkl')
        )
        df_items = add_item_price_group_column(df_items, df_category_price_percentile)
    return df_items


def _run_on_single_batch(cbf, oov_df_items_, inhouse_text_path, attr_file_path, use_item_emb_in_item_tower, feature_names):
    oov_texts = list(oov_df_items_[DatasetField.ITEM_TEXT].fillna(""))

    if inhouse_text_path is None:
        oov_dnn_inputs = torch.from_numpy(_encode_texts(oov_texts)).to(next(cbf.parameters()).device)
    else:
        processor = TokenEmbedProcessor(inhouse_text_path)
        all_embeddings = np.array([processor.run(text) for text in oov_texts])
        oov_dnn_inputs = torch.from_numpy(all_embeddings).to(next(cbf.parameters()).device)

    if use_item_emb_in_item_tower:
        item_embed_vector = cbf.feature_process_modules["item_embed_layer"].get_embed_tensor()
        _, d = item_embed_vector.shape
        oov_item_embs = torch.zeros(len(oov_dnn_inputs), d).to(next(cbf.parameters()).device)
        oov_dnn_inputs = torch.cat([oov_dnn_inputs, oov_item_embs], dim=-1)

    item_side_dnn_inputs = _get_item_side_feat_tensor(oov_df_items_, feature_names, cbf)
    if item_side_dnn_inputs is not None:
        oov_dnn_inputs = torch.cat([oov_dnn_inputs, item_side_dnn_inputs], dim=-1)
    output = cbf.item_tower(oov_dnn_inputs)
    return output


def _get_item_side_feat_tensor(oov_df_items, feature_names, cbf):
    device = next(cbf.parameters()).device
    item_feat_tensors = {}

    if FeatureField.ITEM_MARKET in feature_names:
        item_feat_tensors[FeatureField.ITEM_MARKET] = (
            torch.from_numpy(oov_df_items[FeatureField.ITEM_MARKET].values).to(device).reshape(-1, 1)
        )

    if FeatureField.ITEM_STANDARD_CATEGORY in feature_names:
        item_feat_tensors[FeatureField.ITEM_STANDARD_CATEGORY] = (
            torch.from_numpy(oov_df_items[FeatureField.ITEM_STANDARD_CATEGORY].values).to(device).reshape(-1, 1)
        )

    if FeatureField.ITEM_PRICE_GROUP in feature_names:
        item_feat_tensors[FeatureField.ITEM_PRICE_GROUP] = (
            torch.from_numpy(oov_df_items[FeatureField.ITEM_PRICE_GROUP].values).to(device).reshape(-1, 1)
        )

    if item_feat_tensors:
        item_side_dnn_inputs = cbf.get_dnn_inputs(item_feat_tensors)
        return item_side_dnn_inputs
    else:
        return None


def _save_npy_file(model_path, embeddings: torch.FloatTensor):
    np.save(
        os.path.join(model_path, ITEM_TOWER_OUTPUT_FILENAME),
        embeddings.cpu().detach().numpy(),
    )


def _load_npy_file(model_path):
    return np.load(os.path.join(model_path, ITEM_TOWER_OUTPUT_FILENAME + '.npy'))


def norm_and_save_embedding_as_parquet(model_path):
    item_tower_embed_np = _load_npy_file(model_path)
    encoder = LabelEncoder.from_model_dir(model_path, LabelEncoderPrefix.OUTPUT_ITEM)
    valid_item_ids = encoder.get_valid_item_ids()
    item_tower_embed = torch.nn.functional.normalize(torch.from_numpy(item_tower_embed_np), p=1, dim=1)
    item_tower_embed = item_tower_embed.cpu().numpy()

    data = []
    for sno in valid_item_ids:
        idx = encoder.item_id2index[str(sno)]
        item_embedding = item_tower_embed[idx]
        data.append([sno, item_embedding])

    df_item_embed = pd.DataFrame(data, columns=["sno", "item_embedding"])
    output_path = os.path.join(model_path, ITEM_TOWER_OUTPUT_FILENAME + ".parquet")
    df_item_embed.to_parquet(output_path, index=False)
