"""Multi-Interest 추천 모델 LightningModule.

기존 apps/multi_interest/models/multi_interest_model.py의 학습/평가 로직을
reco-lightning의 LightningModule 패턴으로 마이그레이션.

CBFModel/ComplementModel과의 핵심 차이점:
1. Two-tower 구조가 아닌 self-attention 기반 multi-interest 아키텍처
2. 학습 시 cross-entropy loss (단일 또는 rank learning)
3. 평가 시 interest별 점수를 aggregate하여 최종 점수를 산출
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import lightning as L
from torchmetrics import Metric

from nets.multi_interest.multi_interest_net import (
    ModelInputKey,
    ModelOutputKey,
    MultiInterestNet,
)
from optimizers.linear_warmup import LinearWarmupCosineAnnealingLR
from utils.accuracy import recalls_and_ndcgs_for_ks
from utils.constants import FeatureField


# --- Metrics ---


class LossAccumulator(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, loss):
        self.loss_sum += loss.detach()
        self.total += 1

    def compute(self, reset=True):
        avg_loss = self.loss_sum / self.total
        if reset:
            self.reset()
        return avg_loss


class Accuracy(Metric):
    def __init__(self, top_k=10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0))
        self.add_state("NDCG", default=torch.tensor(0.0))
        self.add_state("Recall", default=torch.tensor(0.0))
        self.top_k = top_k
        self.compute_on_step = False

    def update(self, batch_scores: torch.Tensor, batch_positive_items: torch.Tensor):
        batch_rank_items = batch_scores.argsort(dim=1, descending=True)[:, : self.top_k]
        metrics = recalls_and_ndcgs_for_ks(batch_rank_items, batch_positive_items, [self.top_k])
        self.NDCG += metrics[f"NDCG_{self.top_k}"]
        self.Recall += metrics[f"Recall_{self.top_k}"]
        self.total += 1

    def compute(self):
        scores = {
            f"NDCG_{self.top_k}": self.NDCG / self.total,
            f"Recall_{self.top_k}": self.Recall / self.total,
        }
        self.reset()
        return scores


# --- Loss functions ---


def ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    item_weights: Optional[torch.Tensor] = None,
):
    """Standard cross-entropy loss. 단일 label per sample."""
    logits = logits.view(-1, logits.size(-1))  # (B*T) x V
    labels = labels.view(-1)  # B*T
    loss = F.cross_entropy(logits, labels, ignore_index=0, weight=item_weights)
    return loss


def custom_ce_loss(
    logits: torch.Tensor,
    label_items: torch.Tensor,
    label_scores: torch.Tensor,
    item_weights: Optional[torch.Tensor] = None,
):
    """label이 여러 개이고, 각각의 점수가 다른 경우의 cross_entropy loss.

    F.cross_entropy()는 label이 1개여야 하기 때문에 사용할 수 없다.
    대신, log_softmax + nll_loss를 직접 계산한다.
    """
    label_log_softmax = torch.gather(F.log_softmax(logits, dim=1), dim=1, index=label_items)
    if item_weights is not None:
        label_weights = item_weights[label_items]
        loss = -torch.mean(label_log_softmax * label_scores * label_weights)
    else:
        loss = -torch.mean(label_log_softmax * label_scores)
    return loss


# --- LightningModule ---


class MultiInterestModel(L.LightningModule):
    def __init__(
        self,
        embedding_size: int = 128,
        use_item_bias: bool = True,
        attention_size: int = 8,
        interest_count: int = 4,
        use_rank_learning: bool = False,
        soft_selection: bool = False,
        top_k: int = 10,
        optimizer_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # net은 setup()에서 DataModule의 전처리 결과를 읽어 초기화
        self.net = None
        self.accuracy_metric = None
        self.loss_acc = None
        self._item_train_weights = None

    def setup(self, stage):
        if self.net is not None:
            return

        dm = self.trainer.datamodule

        self.net = MultiInterestNet(
            num_items=dm.num_items,
            num_standard_categories=dm.num_standard_categories,
            embedding_size=self.hparams.embedding_size,
            use_item_bias=self.hparams.use_item_bias,
            attention_size=self.hparams.attention_size,
            interest_count=self.hparams.interest_count,
            max_len=dm.hparams.max_len,
            item_feat_values=dm.item_feat_values,
            soft_selection=self.hparams.soft_selection,
        )

        if dm.item_train_weights is not None:
            self._item_train_weights = torch.FloatTensor(dm.item_train_weights)

        self.accuracy_metric = Accuracy(self.hparams.top_k)
        self.loss_acc = LossAccumulator()

    def on_train_start(self):
        if self._item_train_weights is not None and self._item_train_weights.device != self.device:
            self._item_train_weights = self._item_train_weights.to(self.device)

    def training_step(self, batch, batch_idx):
        out_dict = self.net(batch)

        if self.hparams.use_rank_learning:
            loss = custom_ce_loss(
                out_dict[ModelOutputKey.PREDICTION],
                batch[ModelInputKey.LABEL_ITEMS],
                batch[ModelInputKey.LABEL_SCORES],
                item_weights=self._item_train_weights,
            )
        else:
            loss = ce_loss(
                out_dict[ModelOutputKey.PREDICTION],
                batch[ModelInputKey.POSITIVE_ITEMS],
                item_weights=self._item_train_weights,
            )
        self.loss_acc(loss)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.loss_acc.compute()
        self.log("train/loss", avg_loss)

    def validation_step(self, val_batch, batch_idx):
        input_dict, positive_items = val_batch
        scores = self.compute_score(input_dict, remove_history=True)
        self.accuracy_metric.update(scores, positive_items)

    def compute_score(
        self,
        input_dict: Dict[str, torch.Tensor],
        remove_history: bool = True,
        top_k: int = 1000,
        candidate_list: Optional[list] = None,
        manual_interest_count: Optional[int] = None,
        item_to_display_map: Optional[Dict[int, int]] = None,
        display_decay: float = 0.1,
    ):
        """
        각 interest에서 상품에 대한 점수를 구하고, interest별 점수를 aggregate하여 최종 점수를 산출한다.

        Args:
            input_dict: 모델 입력 텐서 데이터
            remove_history: 유저 히스토리에 있는 상품의 점수를 낮출지 여부
            top_k: 추천 결과 상위에서 대략 몇 개가 필요한지
            candidate_list: 후보 상품 리스트 (서빙 시에만 사용)
            manual_interest_count: 추론 시점에 interest_count를 수동으로 지정 (서빙 시에만 사용)
            item_to_display_map: 상품이 유저에게 노출된 횟수 (서빙 시에만 사용)
            display_decay: 노출된 상품의 점수 감쇠 계수 (서빙 시에만 사용)
        """
        out_dict = self.net(input_dict)
        prediction = out_dict[ModelOutputKey.PREDICTION].softmax(dim=-1)
        interest_score = out_dict[ModelOutputKey.INTEREST_SCORE]

        batch_size, interest_count, item_count = prediction.shape

        # 수동으로 관심사 개수를 지정한 경우
        if manual_interest_count is not None:
            interest_count = min(manual_interest_count, interest_count)
            sorted_idx = torch.argsort(interest_score, descending=True)
            interest_score = interest_score[0][sorted_idx]
            prediction = prediction[0][sorted_idx]

        if top_k > item_count:
            top_k = item_count

        if remove_history:
            history_items = input_dict[FeatureField.CLICK_ITEMS]
            for i in range(batch_size):
                for j in range(interest_count):
                    prediction[i][j][history_items[i]] = -np.inf

        # 후보 상품이 지정되었다면, 후보가 아닌 상품은 prediction 값을 0으로 만든다.
        if candidate_list is not None:
            candidate_items = torch.LongTensor(candidate_list).to(self.device)
            item_candidate_mask = torch.zeros(item_count, device=self.device)
            item_candidate_mask[candidate_items] = 1
            for i in range(batch_size):
                for j in range(interest_count):
                    prediction[i][j] *= item_candidate_mask

        # 노출 페널티
        if item_to_display_map is not None:
            displayed_items = torch.LongTensor(list(item_to_display_map.keys())).to(self.device)
            display_weight = torch.ones(item_count, device=self.device)
            display_weight[displayed_items] = display_decay
            for i in range(batch_size):
                for j in range(interest_count):
                    prediction[i][j] = prediction[i][j] * display_weight

        # 상품의 interest 내 순위에 따라서 점수를 준다. (1 / rank)
        _, ranked_items = prediction.topk(k=top_k)
        default_score_at_rank = 1.0 + torch.arange(ranked_items.shape[2], device=self.device)
        scores = torch.zeros_like(prediction, dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            for j in range(interest_count):
                scores[i][j][ranked_items[i][j]] = interest_score[i][j] / default_score_at_rank

        # 각 interest에서 받은 점수를 합산하여 상품의 최종 점수를 구한다.
        final_scores = torch.sum(scores, dim=1)
        return final_scores

    def on_validation_epoch_end(self):
        dict_ = self.accuracy_metric.compute()
        for k, v in dict_.items():
            self.log(f"val/{k}", v, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        params = self.hparams.optimizer_params or {}
        optimizer_name = params.get("optimizer", "adam").lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 1e-3),
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        lr_scheduler_name = params.get("lr_scheduler", None)
        if lr_scheduler_name is None:
            return optimizer

        if lr_scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=params.get("exponential_lr_gamma", 0.75),
            )
        elif lr_scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=params.get("lr_scheduler_step", 4),
                gamma=params.get("lr_scheduler_gamma", 0.1),
            )
        elif lr_scheduler_name == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=params.get("lr_scheduler_warmup_epochs", 5),
                max_epochs=self.trainer.max_epochs,
                warmup_start_lr=params.get("lr_scheduler_warmup_start_lr", 0.0),
                eta_min=params.get("lr_scheduler_eta_min", 1e-5),
            )
        else:
            raise ValueError(f"Invalid lr_scheduler name: {lr_scheduler_name}")
        return [optimizer], [scheduler]
