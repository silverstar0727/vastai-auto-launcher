"""Multi-Interest 추천 모델의 네트워크 아키텍처.

기존 apps/multi_interest/models/multi_interest_model.py의 순수 nn.Module 부분을 분리.

Self-attention 기반으로 유저 인터랙션 히스토리에서 복수의 관심사(interest) 임베딩을 추출하고,
각 관심사와 전체 상품 임베딩 간의 유사도를 계산하여 추천 점수를 산출한다.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from utils.constants import FeatureField


class ModelInputKey:
    POSITIVE_ITEMS = "positive_items"
    LABEL_ITEMS = "label_items"
    LABEL_SCORES = "label_scores"


class ModelOutputKey:
    PREDICTION = "prediction"
    INTEREST_SCORE = "interest_score"


class MultiInterestNet(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_standard_categories: int,
        embedding_size: int,
        use_item_bias: bool,
        attention_size: int,
        interest_count: int,
        max_len: int,
        item_feat_values: Dict[str, np.ndarray],
        soft_selection: bool = False,
    ):
        """
        Args:
            num_items: padding index를 포함한 item vocabulary size
            num_standard_categories: padding index를 포함한 standard category vocabulary size
            embedding_size: 상품, 이벤트, 시퀀스 위치 임베딩 크기 (크기가 모두 같아야 한다.)
            use_item_bias: 상품 각각에 대한 bias(스칼라 값)를 사용할지 여부
            attention_size: 어텐션의 임베딩 크기
            interest_count: 관심사 개수
            max_len: 유저의 인터랙션 히스토리를 최대 몇 개까지 사용할 것인가
            item_feat_values: 아이템 사이드 피처 (ITEM_STANDARD_CATEGORY 등)
            soft_selection: 학습 시 관심사 선택을 soft (multinomial)로 할지 hard (argmax)로 할지
        """
        super().__init__()
        self.num_items = num_items
        self.num_standard_categories = num_standard_categories
        self.embedding_size = embedding_size
        self.use_item_bias = use_item_bias
        self.attention_size = attention_size
        self.interest_count = interest_count
        self.max_len = max_len
        self.soft_selection = soft_selection
        self.item_feat_values = item_feat_values

        # 상품 임베딩
        self.item_embeddings = nn.Embedding(num_items, embedding_size)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.1 / np.sqrt(embedding_size))

        # 카테고리 임베딩
        self.standard_category_embeddings = nn.Embedding(num_standard_categories, embedding_size)
        nn.init.normal_(self.standard_category_embeddings.weight, mean=0, std=0.1 / np.sqrt(embedding_size))

        # 상품 bias (유저 임베딩과의 유사도로 표현하기 힘든 요소를 상품의 최종 점수에 반영하기 위함)
        if use_item_bias:
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.normal_(self.item_bias.weight, mean=0, std=0.1)

        # 이벤트 임베딩 (unknown=0, click=1, preference=2)
        self.event_embeddings = nn.Embedding(3, embedding_size)
        nn.init.normal_(self.event_embeddings.weight, mean=0, std=0.1 / np.sqrt(embedding_size))

        # 포지션 임베딩
        self.position_embeddings = nn.Embedding(max_len, embedding_size)
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.1 / np.sqrt(embedding_size))

        # Attention layers: W1, W2
        self.W1 = nn.Linear(embedding_size, attention_size)
        self.W2 = nn.Linear(attention_size, interest_count)

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        모델로 상품의 예측값을 계산하여 리턴한다.
        self.training 값(학습 모드인지 추론 모드인지)에 따라 리턴하는 prediction의 shape가 다르다.
        - self.training is True (학습)
            - input_dict에 positive_items가 있어야 하고,
            - 모든 상품에 대해서 단일 점수를 리턴한다. [batch, item]
        - self.training is False (추론)
            - 모든 관심사(interest)에 대해 상품을 점수를 각각 구하여 리턴한다. [batch, interest, item]
        """
        input_dict = self._append_static_features(input_dict)

        history_items = input_dict[FeatureField.CLICK_ITEMS]  # batch, max_len
        history_events = input_dict[FeatureField.EVENT_CODE]  # batch, max_len
        history_standard_categories = input_dict[FeatureField.CLICK_STANDARD_CATEGORIES]  # batch, max_len
        valid_history = (history_items > 0).long()
        batch_size, seq_len = history_items.shape

        # 히스토리에 있는 각 상품의 임베딩
        history_item_emb = self.item_embeddings(history_items)  # batch, max_len, emb
        history_event_emb = self.event_embeddings(history_events)
        history_standard_categories_emb = self.standard_category_embeddings(history_standard_categories)

        # 히스토리 상품 임베딩에, 해당하는 이벤트와 위치(position) 임베딩을 합산
        history_emb_aug = (
            history_item_emb
            + history_event_emb
            + history_standard_categories_emb
            + self.position_embeddings.weight[None, :, :]
        )  # batch, max_len, emb

        # Self-attention
        attention_score = self.W2(self.W1(history_emb_aug).tanh())  # batch, max_len, interest
        attention_score = attention_score.masked_fill(valid_history.unsqueeze(-1) == 0, -np.inf)
        attention_score = attention_score.transpose(-1, -2)  # batch, interest, max_len
        attention_score = (attention_score - attention_score.max()).softmax(dim=-1)
        attention_score = attention_score.masked_fill(torch.isnan(attention_score), 0)

        # 히스토리에 어텐션을 적용하여 관심사 임베딩을 구한다.
        interest_emb = (
            (history_item_emb + history_standard_categories_emb)[:, None, :, :] * attention_score[:, :, :, None]
        ).sum(-2)  # batch, interest, emb

        # (점수를 계산할) 전체 상품 임베딩
        item_emb = self.item_embeddings.weight  # item, emb

        if self.training:
            # 학습할 때는, input_dict에 positive_items가 들어 있어야 한다.
            positive_items = input_dict[ModelInputKey.POSITIVE_ITEMS]
            # 타겟 상품의 임베딩과 가장 비슷한 관심사 임베딩을 선택하고(user_emb), 상품과의 유사도를 구한다.
            target_emb = self.item_embeddings(positive_items).squeeze()
            target_pred = (interest_emb * target_emb[:, None, :]).sum(-1)  # batch, interest
            if self.soft_selection:
                target_prob = torch.softmax(target_pred, dim=-1)  # batch, interest
                idx_select = torch.multinomial(target_prob, 1).flatten()
            else:
                idx_select = target_pred.max(-1)[1]  # batch
            user_emb = interest_emb[torch.arange(batch_size), idx_select, :]  # batch, emb
            prediction = torch.matmul(user_emb, torch.transpose(item_emb, 0, 1))
        else:
            # 추론할 때는, 모든 관심사에 대해서 상품의 유사도 점수를 구한다.
            prediction = torch.matmul(interest_emb, torch.transpose(item_emb, 0, 1))  # batch, interest, item

        if self.use_item_bias:
            prediction += torch.transpose(self.item_bias.weight, 0, 1)

        return {
            ModelOutputKey.PREDICTION: prediction,
            ModelOutputKey.INTEREST_SCORE: torch.linalg.vector_norm(interest_emb, dim=2),
        }

    def _append_static_features(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """클릭한 상품의 카테고리 정보를 input_dict에 추가한다."""
        click_items = input_dict[FeatureField.CLICK_ITEMS]
        item2category = torch.from_numpy(
            self.item_feat_values.get(FeatureField.ITEM_STANDARD_CATEGORY).reshape(-1,)
        ).to(click_items.device)
        input_dict[FeatureField.CLICK_STANDARD_CATEGORIES] = item2category[click_items]
        return input_dict
