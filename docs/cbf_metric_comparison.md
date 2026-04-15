# CBF 기존 메트릭 vs std 메트릭 비교

## 전체 구조

CBF는 Two-tower 구조. `user_tower` → 유저 임베딩, `item_tower` → 아이템 임베딩, 내적으로 점수 계산.

```
user_features → user_tower → user_emb (B, D)
item_features → item_tower → item_emb (N, D)
score = user_emb @ item_emb.T → (B, N)
```

`CBFNet.forward()`의 `item_idxes` 파라미터가 핵심:
- `net(inputs, item_idxes)` → 지정한 아이템들만 점수 계산
- `net(inputs)` → `item_idxes=None` → 전체 아이템(0~num_items) 점수 계산

## 1. 기존 메트릭 (`val/NDCG_50`, `val/Recall_50`)

`src/models/cbf.py` validation_step:

```python
# ① 배치 내 positive 아이템만 추출 (수백 개)
in_batch_items = torch.unique(batch_pos_labels.reshape(-1,))

# ② 이 아이템들에 대해서만 점수 계산
pred_scores = self.net(inputs, in_batch_items)  # (B, ~수백)

# ③ positive가 in_batch_items 내 몇 번째인지 위치 인덱스 계산
targets = [...]  # in_batch_items 내 position index

# ④ 수백 개 후보 중에서 top-50 Recall/NDCG 계산
self.accuracy_metric.update(pred_scores, targets)
```

- 평가 대상: **배치 내 positive 아이템들** (수백 개)
- 히스토리 제거: **없음**
- 경쟁 약함 → 메트릭 높게 나옴
- 용도: **학습 중 빠른 모니터링**

## 2. std 메트릭 (`val/std_NDCG_50`, `val/std_Recall_50`)

```python
# ⑤ 전체 아이템 대상 full ranking
full_scores = self.net(inputs)  # (B, num_items) — 수십만~백만

# ⑥ 유저 클릭 히스토리 아이템을 -inf로 제거
std_scores = full_scores.clone()
click_items = inputs[FeatureField.CLICK_ITEMS]  # (B, max_len)
std_scores.scatter_(1, click_items, float("-inf"))

# ⑦ 전체 아이템 중에서 top-50 Recall/NDCG 계산
self.std_accuracy_metric.update(std_scores, batch_pos_labels)
```

- 평가 대상: **전체 아이템** (수십만~백만)
- 히스토리 제거: **있음** (scatter_ → -inf)
- 전체 아이템 경쟁 → 현실적 메트릭
- 용도: **논문 표준 평가 지표 (실제 성능)**

## 요약

| | 기존 메트릭 | std 메트릭 |
|---|---|---|
| forward 호출 | `net(inputs, in_batch_items)` | `net(inputs)` |
| 평가 범위 | in-batch items (~수백) | 전체 아이템 (~수십만) |
| 히스토리 제거 | X | O (scatter_ → -inf) |
| 로그 키 | `val/NDCG_50` | `val/std_NDCG_50` |
| 용도 | 빠른 모니터링 | 실제 성능 지표 |

early_stopping은 `val/NDCG_50` (기존 메트릭) 기준.
