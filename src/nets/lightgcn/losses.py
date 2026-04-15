"""LightGCN Loss 함수.

learnable temperature를 지원하는 BPR, BCE, CE loss.
temperature는 sigmoid 매핑으로 [temp_min, temp_max] 범위에서 학습된다.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Loss(nn.Module, ABC):
    def __init__(
        self,
        temp: float = 0.5,
        pos_weight: float = 1.0,
        learnable_temp: bool = True,
        temp_min: float = 0.01,
        temp_max: float = 1.0,
    ):
        super().__init__()
        if not (temp_min < temp < temp_max):
            raise ValueError(f"temp={temp} is out of range [{temp_min}, {temp_max}]")
        self._log_temp_min = math.log(temp_min)
        self._log_temp_max = math.log(temp_max)
        p = (math.log(temp) - self._log_temp_min) / (self._log_temp_max - self._log_temp_min)
        eps = 1e-3
        p = min(max(p, eps), 1 - eps)
        raw = math.log(p / (1 - p))
        self._raw = nn.Parameter(torch.tensor(raw), requires_grad=learnable_temp)
        self.pos_weight = pos_weight

    @property
    def temp(self) -> Tensor:
        log_t = self._log_temp_min + (self._log_temp_max - self._log_temp_min) * torch.sigmoid(self._raw)
        return log_t.exp()

    @abstractmethod
    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        ...


class BPRLoss(Loss):
    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        return -F.logsigmoid((pos_scores.unsqueeze(1) - neg_scores) / self.temp).mean()


class BCELoss(Loss):
    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        logits = torch.concat([pos_scores.unsqueeze(1), neg_scores], dim=1) / self.temp
        target = torch.zeros_like(logits)
        target[..., 0] = 1.0
        pw = torch.tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="mean")


class CELoss(Loss):
    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / self.temp
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)


LOSS_REGISTRY: dict[str, type] = {
    "bpr": BPRLoss,
    "bce": BCELoss,
    "ce": CELoss,
}
