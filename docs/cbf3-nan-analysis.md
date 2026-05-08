# CBF3 NaN 재현 및 원인 분석

## 개요

CBF3 모델이 epoch 25 근처에서 train loss가 NaN이 되면서 NDCG가 급격히 떨어지는 현상을 reco-lightning 환경에서 재현하고, 원인을 분석한 문서입니다.

**재현 결과: Epoch 25, Batch 867에서 inf(NaN) 발생** - 원본 보고와 일치

---

## 1. CBF3 설정 (inhouse_kube)

CBF3는 아래와 같은 설정으로 운영됩니다:

| 파라미터 | CBF3 값 | 비고 |
|---|---|---|
| `negative_sampling` | **False** | 180K 전체 softmax |
| `use_attr` | **True** | 이미지 태그 속성 사용 |
| `use_item_emb_in_item_tower` | **True** | item tower에 item embedding 포함 |
| `item_vocabulary_size` | **180,000** | |
| `item_embed_size` | **512** | |
| `batch_size` | **1,024** | |
| `max_items` | **180,000** | |
| `optimizer` | **adam** | |
| `lr` | **0.001** | |
| `weight_decay` | **0.0** | L2 regularization 없음 |
| `lr_scheduler` | **None** | |
| `normalize_outputs` | **False** | L2 norm 미적용 |
| `softmax_temperature` | **1.0** | score 스케일링 없음 |
| `num_epochs` | **40** | |
| `precision` | **AMP (FP16)** | 16-mixed |

---

## 2. 재현 환경

- **Config**: `configs/cbf3-reproduce.yaml`
- **GPU**: NVIDIA RTX A6000 (49GB)
- **데이터**: 3.5억 인터랙션, 180K 아이템, 1.36M train 샘플, 200K eval 샘플
- **모델**: 323M 파라미터 (trainable 96M)

### 코드 변경사항

- `src/datasets/cbf.py`: `use_attr` 지원 추가 (`attr_meta_filename`, `attr_filename` 파라미터 + `_get_feature_process_spec`에 attr 분기 로직)
- `src/models/cbf.py`: 디버그 로깅 추가 (setup, training_step, epoch_end, validation에 shape/값 출력)

---

## 3. 학습 추이

### Score 발산 추이

| Epoch | avg_loss | weight_max | score min | score max | score range |
|---|---|---|---|---|---|
| 0 | 814.2 | 5.49 | -17 | +18 | 35 |
| 1 | 777.0 | 5.51 | -18 | +17 | 35 |
| 2 | 768.4 | 5.53 | -22 | +20 | 42 |
| 3 | 763.6 | 5.55 | -36 | +20 | 56 |
| 4 | 760.2 | 5.56 | -55 | +24 | 79 |
| 5 | 757.7 | 5.61 | -71 | +27 | 98 |
| 6 | 755.6 | 5.57 | -97 | +31 | 128 |
| 7 | 753.9 | 5.76 | -108 | +34 | 142 |
| 8 | 752.3 | 6.39 | -114 | +42 | 156 |
| 9 | 751.1 | 7.01 | -132 | +47 | 179 |
| 10 | 749.9 | 7.59 | -137 | +56 | 193 |
| 11 | 748.8 | 8.15 | -151 | +64 | 215 |
| 12 | 747.9 | 8.70 | -155 | +70 | 225 |
| 13 | 747.0 | 9.22 | -156 | +79 | 235 |
| 14 | 746.3 | 9.73 | -161 | +91 | 252 |
| 15 | 745.6 | 10.22 | -168 | +100 | 268 |
| 16 | 745.0 | 10.69 | -173 | +120 | 293 |
| 17 | 744.4 | 11.16 | -177 | +141 | 318 |
| 18 | 743.8 | 11.63 | -163 | +153 | 316 |
| 19 | 743.3 | 12.06 | -155 | +184 | 339 |
| 20 | 742.9 | 12.50 | -132 | +207 | 339 |
| 21 | 742.5 | 12.91 | -119 | +258 | 377 |
| 22 | 742.1 | 13.33 | -108 | +296 | 404 |
| 23 | 741.8 | 13.72 | -12 | +389 | 401 |
| 24 | 741.7 | 14.09 | +140 | +724 | 584 |
| **25** | - | - | - | **inf** | **NaN** |

### Epoch 25 내 score max 폭발

```
batch    0: max =    787
batch  200: max =  1,055
batch  400: max =  1,304
batch  600: max =  2,338
batch  800: max = 14,488
batch  867: max =    inf  ← FP16 overflow (> 65,504)
```

### Validation NDCG 추이

| Epoch | NDCG_50 | std_NDCG_50 | Recall_50 |
|---|---|---|---|
| 0 | 0.0616 | 0.0824 | 0.0962 |
| 5 | 0.0742 | 0.1065 | 0.1168 |
| 10 | 0.0779 | 0.1148 | 0.1227 |
| 15 | 0.0797 | 0.1186 | 0.1256 |
| 20 | 0.0818 | 0.1225 | 0.1285 |
| 22 | 0.0823 | 0.1234 | 0.1292 |
| 23 | 0.0828 | 0.1233 | 0.1294 |
| 24 | 0.0821 | 0.1209 | 0.1275 |

---

## 4. NaN 발생 메커니즘

### 직접적 원인: FP16 matmul overflow

```python
# 16-mixed 모드에서 이 matmul은 FP16으로 실행
scores = torch.matmul(user_emb, item_emb.T)  # FP16 autocast
```

FP16의 최대 표현값은 **65,504**. Score 값이 이 범위를 넘으면:
```
score > 65,504 → inf (FP16 overflow) → log_softmax(inf) → NaN
```

Epoch 25 batch 867에서 scores 텐서의 180,001개 요소가 모두 inf로 overflow.

### 근본 원인: Two-Tower Dot Product의 양성 피드백 루프

CBF 모델의 score 계산:
```
score_j = user_emb · item_emb_j    (dot product)
```

Cross-entropy loss의 gradient 구조:
```
∂L/∂user_emb  = Σ_j (softmax(s_j) - y_j) · item_emb_j
∂L/∂item_emb_j = (softmax(s_j) - y_j) · user_emb
```

**핵심: user_emb의 gradient는 item_emb에 비례하고, item_emb의 gradient는 user_emb에 비례합니다.**

연립미분방정식으로 보면:
```
d‖user_emb‖/dt  ∝  ‖item_emb‖
d‖item_emb‖/dt  ∝  ‖user_emb‖
```

이는 `d²x/dt² ∝ x` 형태이고, 해가 **지수함수**:

1. `‖item_emb‖` 커짐 → user_emb gradient 커짐 → `‖user_emb‖` 커짐
2. `‖user_emb‖` 커짐 → item_emb gradient 커짐 → `‖item_emb‖` 커짐
3. 1→2→1→2... **상호 증폭 (positive feedback loop)**

### 왜 초반엔 느리고 나중에 폭발하는가

- **초반 (epoch 0~10)**: embedding norm이 작아 상호 증폭 효과 < 학습 signal. 모델이 실제 유의미한 패턴 학습 (loss 814→750)
- **중반 (epoch 10~20)**: norm이 커져서 상호 증폭이 학습 signal을 따라잡기 시작. Score 증가 가속
- **후반 (epoch 20~25)**: 상호 증폭이 완전히 지배. 지수적 폭발 → FP16 overflow

### 모델 아키텍처에서 추적

```
[User Tower]                              [Item Tower]

click_items ──→ item_embedding (512d) ─┐   item_idx ──→ text_embedding ────┐
click_items ──→ text_embedding ────────┤   item_idx ──→ item_embedding ───┤
click_items ──→ attr_vector ───────────┤   item_idx ──→ attr_vector ──────┤
user_age ───→ polynomial_feat ─────────┤   item_market ──→ market_emb ───┤
click_markets ──→ market_emb ──────────┤   item_category ──→ cat_emb ────┤
click_categories ──→ category_emb ─────┤                                  │
                                       │                                  │
              concat (큰 벡터) ←────────┘        concat (큰 벡터) ←────────┘
                    │                                    │
         Linear (bias=False)                  Linear (bias=False)
         BatchNorm1d                          BatchNorm1d
         ELU                                  ELU
                    │                                    │
         Linear → 256d                        Linear → 256d
                    │                                    │
              user_emb (256d)                 item_emb (256d)
              [L2 norm 없음!]                 [L2 norm 없음!]
                    │                                    │
                    └──────── dot product ────────────────┘
                                  │
                         score = u · v  (스칼라)
```

각 레이어가 발산에 기여하는 방식:

1. **Linear(bias=False)**: bias 없이 weight 크기에 순수 비례하는 출력
2. **BatchNorm**: gamma 파라미터가 학습되면서 출력 스케일을 키울 수 있음 (weight_decay=0이므로 gamma 제약 없음)
3. **ELU**: 양수 영역에서 identity (x 그대로 통과) → 큰 값 감쇠 없음
4. **마지막 Linear 후 L2 norm 없음**: `‖output‖`에 상한 없음

만약 `normalize_outputs=True`였다면:
```
user_emb = L2Norm(linear(x))  → ‖u‖ = 1 (항상)
item_emb = L2Norm(linear(x))  → ‖v‖ = 1 (항상)
score = u · v ∈ [-1, +1]      → 절대 overflow 불가
```

Norm이 고정되면 `∂L/∂v ∝ u`에서 `‖u‖=1`이라 gradient가 bounded, 상호 증폭 불가능.

### 발산 과정 수치 추적

```
[epoch 0~10] 정상 학습
  user_emb: ‖u‖ ≈ 5,  item_emb: ‖v‖ ≈ 5
  score = u · v ≈ 5 × 5 × cos(θ) ≈ 18
  gradient ∝ ‖opposite_emb‖ ≈ 5 → 작은 update

[epoch 10~20] 증폭 시작
  user_emb: ‖u‖ ≈ 12,  item_emb: ‖v‖ ≈ 12
  score = u · v ≈ 12 × 12 × cos(θ) ≈ 100
  gradient ∝ ‖opposite_emb‖ ≈ 12 → 2.4배 큰 update → norm 더 커짐

[epoch 20~25] 지수적 폭발
  user_emb: ‖u‖ ≈ 50,  item_emb: ‖v‖ ≈ 50
  score = u · v ≈ 50 × 50 × cos(θ) ≈ 700+
  gradient ∝ ‖opposite_emb‖ ≈ 50 → 10배 큰 update → 폭발
  → score > 65,504 → FP16 inf → NaN
```

---

## 5. Loss가 변하지 않는 이유

### Loss ≈ num_positives × (s_max - s_avg_positive)

`log_softmax`를 전개하면:

```
log_softmax(s_j) = (s_j - s_max) - log(Σ_k exp(s_k - s_max))
```

s_max가 다른 score보다 훨씬 클 때:
```
log_softmax(s_j) ≈ s_j - s_max
```

따라서:
```
loss ≈ Σ_{positive j} (s_max - s_j) = num_pos × (s_max - s_avg_pos)
```

실제 데이터로 검증:
```
loss ~750, num_pos ~82
→ s_max - s_avg_pos ≈ 750/82 ≈ 9.1
```

| Epoch | s_max | s_avg_pos (추정) | 차이 | loss |
|---|---|---|---|---|
| 0 | +18 | ~9 | ~9 | 814 |
| 10 | +56 | ~47 | ~9 | 750 |
| 20 | +207 | ~198 | ~9 | 743 |
| 24 | +724 | ~715 | ~9 | 742 |

절대값이 40배 커져도 **상대적 차이(~9)가 일정** → loss 불변.

**softmax는 상대적 차이에만 의존**하므로, 모든 score에 같은 값을 더해도 loss는 변하지 않습니다 (경제학의 인플레이션과 유사).

### Loss gradient로는 발산을 막을 수 없다

```
∂L/∂s_j = softmax(s_j) - y_j    (항상 [-1, +1] 범위)
```

- 브레이크 힘 (gradient): 일정 (bounded at 1)
- 엔진 힘 (norm 증폭): 지수적 증가

**지수함수는 결국 상수를 이깁니다.**

---

## 6. negative_sampling=False의 기여

### 두 모드의 차이

**Full softmax (`negative_sampling=False`)**:
```python
item_emb = item_tower(ALL 180,001 items)      # 매 step 전체 계산
scores = user_emb @ item_emb.T                 # (1024, 180001)
```

**In-batch negative sampling (`negative_sampling=True`)**:
```python
in_batch_items = unique(batch_pos_labels)       # ~2000-5000개
item_emb = item_tower(in_batch_items)           # (2000-5000, 256)
scores = user_emb @ item_emb.T                  # (1024, 2000-5000)
```

### 기여 경로 1: gradient 업데이트 빈도

| | Full softmax | Negative sampling |
|---|---|---|
| 아이템당 gradient 업데이트 | **1,330회/epoch** (매 batch) | **~20-50회/epoch** (등장 시만) |
| 양성 피드백 루프 | 매 step 연속 작동 (연속 복리) | 간헐적 작동 (단리) |

Full softmax에서는 모든 아이템이 매 step gradient를 받아 양성 피드백 루프가 **연속적으로** 돌아감. Negative sampling에서는 **간헐적**으로만 돌아감.

### 기여 경로 2: softmax 집중도

180K-way softmax vs 3000-way softmax에서 "지배적 아이템"의 영향력 차이:

**180K-way softmax**: embedding norm이 큰 아이템이 `softmax ≈ 1.0` → 확률 완전 독점 → user_emb gradient를 지배 → 모든 user embedding에 공통 bias 생성 → 전체 score 동반 상승

**3000-way softmax**: 지배적 아이템이 배치에 없을 수 있음. 있더라도 3000개 중 하나라 영향력 분산.

### 비유

| | Full softmax | Negative sampling |
|---|---|---|
| 비유 | 전교생 180K명이 **매일** 동일한 시험에서 경쟁 | 매일 **랜덤 30명**끼리 경쟁 |
| 1등 효과 | 1등이 **매일** 전교생의 자원을 가져감 | 1등이 가끔만 참여, 영향력 제한 |
| 결과 | 부익부 빈익빈 → 독점 → 폭발 | 기회 분산 → 안정 |

---

## 7. NDCG가 NaN보다 먼저 떨어지는 이유

### 시간순서

```
[1단계] Epoch 0~20: embedding 천천히 성장
  → 랭킹은 아직 content 기반으로 정상 작동
  → NDCG 꾸준히 상승 (0.062 → 0.082)

[2단계] Epoch 22~24: 소수 아이템의 norm이 지배적
  → Top-50이 norm 순서로 수렴 (개인화 상실)
  → NDCG 정체 또는 소폭 하락 (0.082 → 0.082)
  → scores는 아직 FP16 범위 안 → 수치적으로는 정상

[3단계] Epoch 25: score가 FP16 max(65504)를 초과
  → inf → NaN → 모든 메트릭 붕괴
```

**2단계와 3단계 사이에 갭이 존재하는 이유**: 랭킹 품질 열화(NDCG 하락)는 embedding norm **비율**이 왜곡되면 발생하고, 수치 overflow(NaN)는 **절대값**이 FP16 한계를 넘으면 발생합니다. 비율 왜곡이 절대값 한계보다 먼저 옵니다.

---

## 8. CBF3에서 NaN을 방지하는 방법들

| 방법 | 효과 | 양성 피드백 루프에 대한 작용 |
|---|---|---|
| `normalize_outputs=True` | `‖emb‖ = 1`로 고정 | **루프 자체를 차단** (norm 증가 불가) |
| `weight_decay > 0` | L2 regularization | **평형점 생성** (`d‖e‖/dt ∝ ‖u‖ - λ‖e‖`) |
| `softmax_temperature < 1` | score 스케일 축소 | gradient 크기 제한 |
| `negative_sampling=True` | 소수 아이템만 경쟁 | 루프 실행 빈도 대폭 감소 |
| `mean_loss=True` | positive 수로 나눔 | gradient 크기 제한 |
| `precision=32` 또는 `bf16` | 더 넓은 수치 범위 | overflow 시점 지연 (근본 해결 아님) |

가장 효과적인 조합은 `normalize_outputs=True` + `softmax_temperature=0.05` (현재 `cbf.yaml`의 설정)으로, 이는 embedding norm을 1로 고정하여 양성 피드백 루프 자체를 원천 차단합니다.

---

## 9. Two-Tower 논문과의 비교: 논문의 문제인가, 구현의 문제인가?

### 관련 핵심 논문들

Two-Tower 추천 모델의 계보는 크게 두 논문에서 시작합니다:

**1. YouTube DNN (Covington et al., 2016)**
- Two-Tower의 원조격. User DNN + Item embedding으로 inner product 계산
- **핵심: sampled softmax 사용** — 수천 개의 negative sample만 사용하고, 전체 softmax는 쓰지 않음
- 논문 원문: *"We sample several thousands of negatives and minimize the softmax loss for the true label"*
- 또한 유저별 고정 수의 학습 샘플을 사용하여 활발한 유저가 loss를 지배하지 않도록 함

**2. Google Two-Tower (Yi et al., 2019) — "Sampling-Bias-Corrected Neural Modeling"**
- Two-Tower라는 이름을 명시적으로 사용한 논문
- YouTube 추천에 실 배포
- **핵심: in-batch negative sampling + bias correction**
- 전체 softmax가 아니라, 배치 내 다른 아이템을 negative로 사용
- 인기 아이템이 negative에 과대표현되는 문제를 streaming frequency estimation으로 보정

### 논문들의 설계 vs CBF3

| 설계 요소 | YouTube DNN (2016) | Google Two-Tower (2019) | CBF3 |
|---|---|---|---|
| **Softmax 범위** | sampled (수천 개) | in-batch (~배치크기) | **전체 180K** |
| **Negative sampling** | O (수천 개) | O (in-batch) | **X** |
| **Score 함수** | inner product | inner product | inner product |
| **Norm 정규화** | 명시 안됨 | 명시 안됨 | **없음** |
| **Weight decay** | 논문에 미기술 | 논문에 미기술 | **0.0** |

**핵심 발견: 원조 논문들은 애초에 전체 softmax를 쓰지 않았습니다.**

YouTube DNN은 수천 개의 sampled negative, Google Two-Tower는 in-batch negative를 사용했습니다. 이것은 단순히 "계산 비용 절감"이 아니라, **학습 안정성에도 결정적인 역할**을 합니다. Negative sampling은 양성 피드백 루프의 실행 빈도를 ~30-60배 줄여서 embedding 발산을 억제합니다.

### Contrastive Learning 관점에서의 분석

Two-Tower 모델은 사실 contrastive learning의 한 형태입니다. 이 분야에서는 이미 embedding norm 문제가 잘 알려져 있습니다:

**Wang & Isola (2020) — "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"**
- Contrastive loss가 최적화하는 두 가지 성질: alignment(positive pair 유사도)와 uniformity(embedding 분포 균일성)
- **핵심: 이 분석은 hypersphere 위에서**, 즉 **L2 normalized embedding**을 전제로 함
- Norm이 자유로운 상태에서는 이론적 보장이 성립하지 않음

**"Is Cosine-Similarity of Embeddings Really About Similarity?" (2024)**
- Dot product vs cosine similarity의 차이를 수학적으로 분석
- **regularization 없이 학습된 embedding에서 cosine similarity는 임의적이고 무의미한 결과를 줄 수 있음**
- 즉, **norm 제약 없는 dot product 학습은 이론적으로도 불안정**

**"Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics" (2023)**
- Feature normalization이 representation collapse를 방지하는 핵심 역할
- Cosine loss (normalized)는 regularization 강도와 무관하게 collapse하지 않음
- **Unnormalized dot product는 이런 보장이 없음**

### 평가: 구현(CBF3)의 문제

**원조 논문의 문제가 아니라 CBF3 구현의 문제입니다.** 이유:

1. **원조 논문들은 전체 softmax를 쓰지 않았습니다.** YouTube DNN과 Google Two-Tower 모두 sampled/in-batch negative sampling을 사용. CBF3의 `negative_sampling=False`는 원조 논문의 설계에서 벗어난 선택.

2. **업계 표준은 normalization 또는 temperature scaling을 포함합니다.** Contrastive learning 문헌에서 NT-Xent (Normalized Temperature-scaled Cross Entropy)가 표준인 것은, L2 norm + temperature가 학습 안정성에 필수적이기 때문. CBF3는 둘 다 없음.

3. **논문들은 이 문제를 "언급하지 않은 것"이 아니라 "설계로 회피"한 것입니다.** Negative sampling을 쓰면 embedding 발산이 실질적으로 발생하지 않으므로, 이 문제를 경험할 기회 자체가 없음.

4. **CBF3의 설정 조합이 독특합니다:**
   - `negative_sampling=False` (전체 180K softmax) — 원조 논문에 없는 선택
   - `normalize_outputs=False` — norm 제약 없음
   - `weight_decay=0.0` — regularization 없음
   - `softmax_temperature=1.0` — scaling 없음

   이 4가지가 동시에 꺼진 상태는 논문에서 다룬 적이 없는 조합. 각각은 단독으로 문제가 안 될 수 있지만, **4개가 모두 꺼지면** 양성 피드백 루프를 막을 장치가 전혀 없게 됨.

### 한 줄 결론

**Two-Tower 논문들은 negative sampling을 사용하여 이 문제를 구조적으로 회피했고, CBF3는 `negative_sampling=False` + 정규화 없음이라는 논문에 없는 조합을 사용하여 embedding 발산 문제에 노출된 것입니다.**

---

## 10. 생성된 파일

| 파일 | 설명 |
|---|---|
| `configs/cbf3-reproduce.yaml` | CBF3 동일 설정 config |
| `src/datasets/cbf.py` | `use_attr` 지원 추가 |
| `src/models/cbf.py` | 디버그 로깅 추가 |

---

## References

- [Deep Neural Networks for YouTube Recommendations (Covington et al., 2016)](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf)
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (Yi et al., 2019)](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere (Wang & Isola, 2020)](https://arxiv.org/abs/2005.10242)
- [Is Cosine-Similarity of Embeddings Really About Similarity? (2024)](https://arxiv.org/abs/2403.05440)
- [Feature Normalization Prevents Collapse of Non-contrastive Learning Dynamics (2023)](https://arxiv.org/abs/2309.16109)
- [Two-Tower Networks and Negative Sampling in Recommender Systems](https://towardsdatascience.com/two-tower-networks-and-negative-sampling-in-recommender-systems-fdc88411601b/)
- [The Two-Tower Model for Recommendation Systems: A Deep Dive](https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive)
- [Dot Product or Cosine? (Michael Roizner)](https://roizner.medium.com/dot-product-or-cosine-55cd5b22a87c)
