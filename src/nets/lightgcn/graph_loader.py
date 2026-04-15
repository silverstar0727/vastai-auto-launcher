"""LightGCN Graph 구축 및 캐싱.

Bipartite user-item 그래프를 CSR 포맷의 정규화된 인접 행렬로 구축한다.
정규화: A_norm[u→v] = (deg[u] * deg[v])^(-0.5) (symmetric normalization)
"""

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class GraphLoader:
    def __init__(
        self,
        events: pd.DataFrame,
        n_users: int,
        n_items: int,
    ):
        self._n_users = n_users + 1  # 0: unknown user
        self._n_items = n_items + 1
        self._N = self._n_users + self._n_items

        self._users = torch.from_numpy(events["user_id"].to_numpy(np.int32))
        self._items = torch.from_numpy(events["item_id"].to_numpy(np.int32))

        self._cached: Tensor | None = None
        self._cached_device = None

    @property
    def n_nodes(self) -> int:
        return self._N

    @property
    def n_edges(self) -> int:
        return self._users.size(0) * 2

    def load(self, device) -> Tensor:
        device = torch.device(device)
        if self._cached is None or self._cached_device != device:
            self._cached_device = device
            rows, cols = self._to_bidirectional(self._users, self._items)
            self._cached = self._build_normalized_csr(rows, cols).to(device)
        return self._cached

    def _to_bidirectional(self, users: Tensor, items: Tensor) -> tuple[Tensor, Tensor]:
        items_offset = items + self._n_users
        rows = torch.cat([users, items_offset])
        cols = torch.cat([items_offset, users])
        return rows, cols

    def _build_normalized_csr(self, rows: Tensor, cols: Tensor) -> Tensor:
        sort_idx = rows.argsort()
        rows, cols = rows[sort_idx], cols[sort_idx]

        degree = torch.bincount(rows, minlength=self._N).int()
        inv_sqrt = torch.zeros_like(degree, dtype=torch.float32)
        pos_mask = degree > 0
        inv_sqrt[pos_mask] = degree[pos_mask].pow(-0.5)

        values = inv_sqrt[rows] * inv_sqrt[cols]

        crow = torch.zeros(self._N + 1, dtype=torch.int32)
        crow[1:] = degree.cumsum(0)

        return torch.sparse_csr_tensor(crow, cols, values, (self._N, self._N))
