"""LightGCN 평가 메트릭.

Recall@k, NDCG@k, Coverage@k를 계산한다.
CBF/Complement 모델의 Accuracy와 달리 single positive target을 사용한다.
"""

import torch
from torch import Tensor


class AccuracyAndCoverage:
    def __init__(self, num_items: int, top_ks: list[int] = [10, 20, 50]):
        self.top_ks = sorted(top_ks)
        self.max_k = self.top_ks[-1]
        self.num_items = num_items
        self._total = 0
        self._recall_sum = {k: 0.0 for k in self.top_ks}
        self._ndcg_sum = {k: 0.0 for k in self.top_ks}
        self._cover_masks: dict[int, Tensor] = {}

        position = torch.arange(2, 2 + self.max_k, dtype=torch.float)
        self._weights = 1.0 / torch.log2(position)

    def update(self, scores: Tensor, target: Tensor) -> None:
        """
        Args:
            scores: (B, num_items+1) 각 아이템에 대한 점수
            target: (B,) positive 아이템 인덱스 (1개)
        """
        _, topk_idx = scores.topk(self.max_k, dim=1)
        hits = topk_idx == target.unsqueeze(1)
        w = self._weights.to(scores.device)

        for k in self.top_ks:
            hits_k = hits[:, :k]
            self._recall_sum[k] += hits_k.any(dim=1).sum().item()
            self._ndcg_sum[k] += (hits_k * w[:k]).sum().item()

            if k not in self._cover_masks:
                self._cover_masks[k] = torch.zeros(self.num_items + 1, dtype=torch.bool, device=scores.device)
            self._cover_masks[k][topk_idx[:, :k].reshape(-1)] = True

        self._total += scores.size(0)

    def compute(self) -> dict[str, float]:
        result = {}
        for k in self.top_ks:
            coverage_count = self._cover_masks[k].sum().item() if k in self._cover_masks else 0
            result[f"NDCG_{k}"] = self._ndcg_sum[k] / self._total
            result[f"Recall_{k}"] = self._recall_sum[k] / self._total
            result[f"Coverage_{k}"] = coverage_count / self.num_items
        self.reset()
        return result

    def reset(self) -> None:
        self._total = 0
        for k in self.top_ks:
            self._recall_sum[k] = 0.0
            self._ndcg_sum[k] = 0.0
        for mask in self._cover_masks.values():
            mask.zero_()
