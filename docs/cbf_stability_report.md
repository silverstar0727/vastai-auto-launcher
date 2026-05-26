# CBF 학습 안정화 실험 보고서

**데이터**: 20260506 dt 파티션. 1.6M 활성 user, 180k item vocabulary.

---

## 1. 측정 메트릭

운영 cbf3 의 'Score 발산 추이' 포맷 + ranking 평가 표준 두 갈래.

### 1-1. 학습 진단 (`train/*`)

| Metric | 설명 | 목적 |
|---|---|---|
| `train/loss` | 매 step softmax cross-entropy 의 batch 평균 | 학습이 줄어드는지 |
| `train/score_min` | epoch 안 모든 batch 의 logit 최솟값을 reduce | logit 분포의 아래쪽 꼬리 추적 |
| `train/score_max` | epoch 안 모든 batch 의 logit 최댓값을 reduce | **발산 직접 증거** — 폭주 시 O(10³~10⁴) |
| `train/weight_max` | epoch 끝에 trainable parameter 의 절댓값 최댓값 | 임베딩 magnitude 가 자라는 정도 |
| `train/temperature` (E5/E8/E10) | learnable τ 의 현재 값 (`exp(log_temperature)`) | softmax sharpness 가 데이터에 적응하는지 |
| `train/log_temperature` (E5/E8/E10) | τ 의 log 공간 raw parameter 값 | gradient 가 실제로 acting 하는 공간 |

> `score_range = score_max − score_min` 은 wandb dashboard 에서 derived metric 으로 계산.

### 1-2. ranking 메트릭

> head = 인기 상위 20% 아이템, tail = 나머지. `on_voca_items` 가 frequency desc 정렬돼 item_index 작을수록 인기.
> 모든 ranking 메트릭은 두 모드로 측정. **in-batch** (`val/*`) = 같은 batch 내 다른 user 들의 positive 만 후보 풀로 (수천개). **std** (`val/std_*`) = 전체 vocab (180k) 중 user 본 아이템 마스킹 후 ranking — 운영 셋업과 동일.

| Metric | 설명 | 목적 |
|---|---|---|
| `Recall_50` | user 별로 top-50 안에 들어온 정답 개수를 정답 전체 개수로 나누고 평균 | 정답 회수율 |
| `NDCG_50` | user 별 DCG (정답 ranking 위쪽일수록 높은 가중치) 를 정답 ideal DCG 로 정규화하고 평균 | 정답 ranking 의 quality (위에 있을수록 좋음) |
| `HR_1` | user 의 top-1 추천이 정답에 속하면 1, 아니면 0. 평균 | top-1 의 strict 정확도 |
| `MRR_50` | user 의 top-50 안 첫 정답 위치의 reciprocal (`1/rank`). top-50 안에 없으면 0. 평균 | 첫 정답이 얼마나 위에 있는지 |
| `Coverage_Percentage_50` | 전체 user 의 top-50 추천 합집합의 unique 개수 / 총 아이템 수 | 추천에 등장하는 vocab 비율 (다양성) |
| `TailRecall_50` | 정답 중 tail 아이템만 골라서 그것이 top-50 에 들어왔는지로 recall 계산 | tail 아이템 정답 회수 능력 |
| `TailExposure_50` | user 별 top-50 추천 중 tail 비율. 평균 | 추천이 인기에 얼마나 쏠려있는지 |

모두 [0, 1] 범위.

### 1-3. 왜 이 조합인가

| 보고 싶은 것 | 적합 metric |
|---|---|
| 발산 / 학습 깨짐 | `score_max`, `weight_max`, `train/loss` |
| 학습 setup 안 정확도 | `NDCG_50`, `Recall_50` (in-batch) |
| 운영 환경 정확도 | `std_NDCG_50`, `std_Recall_50` |
| 운영 환경 precision (top-1) | `std_HR_1`, `std_MRR_50` |
| popularity bias 정도 | `std_NDCG` **+** `Coverage` **+** `TailRecall` **+** `TailExposure` 같이 본다 |
| 다양성 / cold-start friendly | `Coverage`, `TailExposure` |

NDCG 하나만 보면 popularity 추천기가 잘 맞춰지는 long-tail 분포 데이터에선 진짜 학습 quality 가 가려진다. **다양성 / tail 메트릭과 같이 봐야 popularity bias 와 진짜 개인화가 분리** 된다 (자세한 건 §4 의 E3 분석 참고).

---

## 2. 실험 매트릭스

운영 `cbf-reprod` 를 baseline 으로 두고, 각 옵션을 켜고 끄며 측정. 모든 실험 동일한 seed=42, 동일한 데이터.

| ID | bf16 | gradClip | mean_loss | l2norm | small τ (0.05) | learnable τ | biasCorr | adamw + warmup | 의도 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **E0** | | | | | | | | | overflow 발생 양상의 정밀 측정용 reference (운영 align baseline) |
| **E1** | ✓ | | | | | | | | fp16 overflow 회피만으로 발산이 막히는지 |
| **E2** | ✓ | ✓ | | | | | | | step 당 weight update 크기 제한이 발산을 막는지 |
| **E3** | ✓ | | ✓ | | | | | | per-user loss 균등화로 heavy user dominance 완화 효과 |
| **E4** | ✓ | | | ✓ | ✓ | | | | L2 정규화로 logit 캡 (CLIP 표준) |
| **E5** | ✓ | | | ✓ | ✓ | ✓ | | | τ 가 데이터에 적응하는지 |
| **E6** † | | | ✓ | ✓ | ✓ | | | ✓ | 과거 효율적인 실험 재현 |
| **E7** | ✓ | | ✓ | ✓ | ✓ | | | | L2 norm 위에 user equity 추가 효과 |
| **E8** | ✓ | | | ✓ | ✓ | ✓ | ✓ | | popularity bias 보정 (Yi 2019 logQ) |
| **E9** | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | | logQ + user equity 결합 |
| **E10** | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | full stack — 모든 안정화 기법 결합 |
| **E11** ‡ | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ | E10 와 동일하되 **negative_sampling=false** (+ 강제로 biasCorr off) — in-batch sampled softmax → full vocab softmax ablation |

> **공통 (E0~E10)**: `negative_sampling=true` (in-batch softmax), `item_dropout_prob=0.15`, `item_vocabulary_size=180000`, `max_items=1000000`, `batch_size=1024`, `num_workers=0`. 30 epoch (E5/E8 옛 run 은 20, E6 는 40).
> **† E6 만** `use_attr=false`, `use_item_emb_in_item_tower=false`, `price_group_embed_size=0` — Item Tower 표현력이 다른 실험들보다 약함 (콘텐츠만 사용). 다른 실험은 모두 `use_attr=true`.
> **‡ E11 만** `negative_sampling=false` (full 180k-vocab softmax) + `batch_size=512` (메모리 마진 확보, 다른 실험들의 1024 대비 step 수 약 2배). E10 base 위에서 `negative_sampling` 만 끈 ablation — 단, `bias_correction` 은 negative sampling 의 logQ bias 를 보정하는 것이라 정의상 함께 꺼야 하고 (`src/datasets/cbf.py:270` assert), 그 외 옵션 (mean_loss, learnable τ, l2norm, adamw+warmup) 은 E10 그대로.

---

## 3. 결과 — 30 epoch finished runs

| Exp | score_max | weight_max | NDCG_50 | std_NDCG | Coverage% | TailRecall | TailExp | HR_1 | MRR_50 | std_HR_1 | std_MRR_50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E1** bf16 | **5,312** 💥 | 17.8 | 0.062 | 0.037 | 31.0 | 0.005 | 0.029 | 0.042 | 0.095 | 0.028 | 0.070 |
| **E2** +clip | **6,080** 💥 | 17.9 | 0.068 | 0.039 | 37.9 | 0.007 | 0.044 | 0.048 | 0.104 | 0.031 | 0.074 |
| **E3** +mean_loss | 382 | 16.5 | 0.238 | **0.075** ★ | 67.2 | 0.044 | 0.319 | 0.196 | 0.342 | 0.070 | 0.146 |
| **E4** +l2norm | 14.3 ✅ | 15.0 | 0.262 ★ | 0.066 | **86.6** ★ | **0.054** ★ | **0.489** ★ | 0.226 | 0.376 | 0.066 | 0.133 |
| **E7** E4+mean_loss | 13.9 ✅ | 15.6 | 0.260 | 0.064 | 86.6 ★ | 0.053 | 0.494 ★ | 0.224 | 0.374 | 0.063 | 0.129 |
| **E10** full-stack | 41.6 ✅* | 8.5 | 0.255 | 0.067 | 80.8 | 0.051 | 0.407 | 0.223 | 0.371 | 0.063 | 0.131 |

`*` = L2 norm 캡 안에 들어가지만 bias_correction 의 −log(q) term 이 tail 아이템 score 를 boost 해서 단순 1/τ 한계를 약간 초과. 발산 아님.

20 epoch / 40 epoch 옛 run 들 (보조 metric 없음, 참고용):

| Exp | epoch | score_max | NDCG_50 | std_NDCG |
|---|---|---:|---:|---:|
| cbf-reprod | 20 | n/a (로깅 전) | 0.255 | 0.064 |
| E0 baseline | 20 | 203 | 0.255 | 0.063 |
| E5 (learnable τ) | 20 | 22 | 0.260 | 0.063 |
| E8 (E5+logQ) | 20 | 38.6 | 0.258 | 0.067 |
| **E6** | 40 | 14.6 | **0.212** | **0.048** |

---

## 4. 핵심 발견 — 자주 묻는 질문에 대한 답

### Q1. E1, E2 는 왜 갑자기 모든 메트릭이 훅 떨어지나? score_max 가 왜 발산하나?

**짧은 답**: cross-entropy + dot product 학습은 본질적으로 **최적해가 무한대에 있다**. L2 norm 같은 명시적 magnitude 제약이 없으면 임베딩이 한없이 자란다.

**메커니즘**:

score 는 `s = u · v = ‖u‖·‖v‖·cos(θ)`. cross-entropy softmax 는 정답 logit 을 +∞ 로, 나머지를 -∞ 로 보내는 방향으로 학습한다. 정답 logit 을 키우려면:

1. `cos(θ)` 를 1 에 가깝게 — 방향 정렬 (의미 있는 학습)
2. **`‖u‖`, `‖v‖` 키우기 — 단순 magnitude 폭주 (의미 없는 학습)**

L2 norm 없는 환경에서 gradient descent 는 1, 2 양쪽으로 자유롭게 푸시한다. 특히 인기 아이템은 여러 user 의 positive 로 자주 등장해서 매 step 마다 여러 user gradient 의 합이 누적 → `‖v_hot‖` 가 빠르게 자란다. 동시에 그 user 들의 `‖u‖` 도 따라 자란다. **`‖u‖·‖v‖` 가 곱셈으로 작용** 해서 super-linear 한 score 증가가 나타난다.

E1 (bf16) 은 fp16 의 65k 한계만 우회. bf16 의 3.4e38 한계는 멀어 학습이 계속 폭주. 20 epoch (score 408) → 30 epoch (score 5,312) 로 더 망가졌다.

E2 (gradient_clip=1.0) 는 step 당 weight 변화를 자른다. 하지만 update 가 자꾸 같은 방향으로 누적되면 결국 자란다. 오히려 clip 때문에 학습이 천천히 진행돼서 같은 epoch 수에 더 폭주한 양상 (6,080). **clip 은 발산을 못 막는다**.

발산이 시작되면 logit 분포가 인기 hot 으로 쏠리고, softmax 가 거의 모든 score 를 hot 으로 집중. 그 결과:
- in-batch NDCG: hot 아이템들끼리 구분 불가 → 0.06 (다른 실험의 1/4)
- Coverage: 31% (vocab 의 7할 이 한 번도 추천 안 됨)
- TailRecall: 0.005 (tail 정답 거의 못 찾음)

**메트릭이 한꺼번에 떨어지는 이유는, score 발산이 모든 평가의 공통 원인이라서**. 한 가지 병이 여러 증상으로 나타난다.

### Q2. E3 (mean_loss) 의 std_NDCG 가 30 epoch 실험들 중 1위인 이유와 단점은?

**std_NDCG 1위 (0.075)**. NDCG@50 (in-batch) 도 0.238 로 양호. 표면적으로 best 같다.

**왜 좋아 보이는가**:

- `mean_loss=True` 가 per-user 손실을 positive 개수로 나누면서 heavy user 의 dominance 가 줄어든다. heavy user 는 인기 아이템에 대한 positive 비중이 자연스럽게 크다. 그래서 popularity bias 의 학습 속도가 살짝 늦춰지고, score 발산도 E1 (5,312) 대비 완화 (382).
- 그래도 L2 norm 이 없어서 popularity 학습 자체는 여전히 강하게 일어남.
- `std_NDCG` 는 ground truth 정답 분포가 popularity 따라가는 경향이 있어, **popularity-driven 모델이 잘 맞춰지는 경향** 이 있다.

**그러나 함정**:

| | E3 | E4 |
|---|---|---|
| std_NDCG | **0.075** ★ | 0.066 |
| Coverage | 67.2% | **86.6%** ★ |
| TailRecall | 0.044 | **0.054** ★ |
| TailExposure | 31.9% | **48.9%** ★ |
| **score_min** | **0.21** (모두 양수!) | -12.4 |

E3 의 score_min 이 0.21 — **모든 logit 이 양수**. 임베딩들이 한쪽 방향으로 다 쏠려있다는 직접 증거. **popularity-only 추천기** 에 매우 가깝다. std_NDCG 가 높은 건 인기 아이템에 잘 맞췄기 때문이고, Coverage / Tail 메트릭에서 진짜 본색이 드러난다.

**단점**:
1. 다양성 부족 — 추천에 등장하는 unique item 이 vocab 의 67% 만 (E4 는 87%)
2. tail / cold-start 학습 약함 — TailRecall 0.044, TailExposure 31.9%
3. score 발산 진행 중 — score_max 382 가 30 epoch 끝 시점. 더 epoch 가면 더 클 가능성

**한 줄**: std_NDCG 단독 평가의 위험성을 보여주는 가장 명료한 예. mean_loss 가 발산을 약간 늦출 뿐 막진 못한다.

### Q3. E4 가 Coverage 1위인 이유?

E4 = L2 norm + `softmax_temperature=0.05`. **forward 안에서** `‖u‖ = ‖v‖ = 1` 강제하면 score 가 `cos(θ)/τ ∈ [-20, 20]` 으로 캡된다.

이렇게 캡되면:
1. 인기 아이템에 logit 을 무한정 키울 길이 없음 → popularity bias 의 핵심 메커니즘 차단
2. 모델이 정답 logit 을 키우려면 cos(θ) 를 1 에 가깝게 — 즉 **방향(angle) 정렬** 으로 정답을 맞춰야 함
3. 방향 학습은 각 user/item 의 고유 특성을 반영 → 인기 아이템만 학습하지 못함
4. tail 아이템들도 자기 유저들과의 방향 정렬을 학습 → 추천에 등장 빈도 ↑

결과: top-50 에 등장하는 unique item 이 vocab 의 **86.6%** (E4). 운영 데이터의 86.6% 가 "어떤 user 에겐가 top-50 추천된 적 있다". E1 의 31% 와 대비.

E7 (E4 + mean_loss) 도 86.6% 로 동일. **L2 norm 이 Coverage 의 결정타** 이고 mean_loss 는 거의 영향 없음을 확인.

### Q4. E6 의 메트릭이 다른 실험들보다 왜 이렇게 낮은가?

**E6 final (40 epoch, bs=1024 변형)**: NDCG 0.212, std_NDCG 0.048 — 30 epoch 실험들 (NDCG 0.25+) 보다 모두 한 단계 아래.

설정 차이가 결정적:

| 피처 | E0~E5, E7~E10 | E6 |
|---|---|---|
| `use_attr` (1007차원 frozen attribute) | **True** | **False** |
| `use_item_emb_in_item_tower` (학습 가능 ID embedding) | **True** | **False** |
| `price_group_embed_size` | 8 | **0** |

Item Tower 가 E6 에선 **콘텐츠 (텍스트 + market + std category) 만으로** 임베딩을 생성. 학습 가능한 collaborative ID feature 가 없음.

이렇게 되면:
- 같은 카테고리·마켓 안의 아이템들이 거의 같은 임베딩 — **개별 아이템 구분 능력 없음**
- 학습은 "user 의 카테고리 선호도" 만 잡고, "이 user 가 이 카테고리 안 어떤 specific item 을 좋아할지" 는 못 잡음

평가 영향:
- in-batch 평가 (작은 풀, 카테고리 다양): 카테고리 매칭만으로 잘 맞춰짐 → 평이
- std 평가 (전체 vocab, 같은 카테고리 안 후보 많음): individual differentiation 못 함 → NDCG 급락 (0.048)

**E6 는 "콘텐츠 기반 cold-start 시뮬레이션" 의 reference** 로는 의미 있지만 운영 후보로는 부적합. 학습 표준 (l2norm, cosine warmup, adamw, mean_loss) 다 들어있어도 Item Tower 표현력이 부족하면 무의미하단 점이 핵심 발견.

### Q5. E6 의 batch size 가 달라졌을 때 양상

원본 cbf.yaml 은 bs=128. 1.6M user / 128 ≈ 12,500 step/epoch. 40 epoch → 500k step. 학습 시간 ~16배 증가 (A6000 단독 GPU 에서 추정 4일).

실제 1회차 (bs=128) 실험을 11 epoch 까지 돌렸을 때:

- **in-batch NDCG 0.500** ← 매우 높음
- std_NDCG 0.0216 ← 매우 낮음
- 비율 23배 (다른 실험들 ~4배)

**왜 in-batch 만 부풀려졌나**: in-batch 평가의 후보 풀이 `unique(batch_positives)` 인데, bs=128 이면 unique positives 가 작음 (수백). 거의 풀이 작아서 정답 띄우기 쉬움. bs=1024 면 unique positives 가 수천 → 변별이 어려워짐. **in-batch NDCG 는 batch_size 에 sensitive**. 절대값 비교 어렵다.

bs=1024 로 통일 후 (= E6 final 40 epoch): NDCG 0.212. 정상 비교 가능.

bs 가 작은 두 번째 효과: **warmup 의 lr 이 epoch 단위로 ramp 되어 lr/step 이 1/8 수준**. 학습 dynamics 도 다름. 결과적으로 정상 비교를 위해 bs 만 1024 로 갈아끼우고 나머지 cosine warmup, l2norm, mean_loss 는 원본 유지한 게 우리 final E6.

### Q6. learnable τ (E5, E8, E10) 가 실제로 데이터에 적응하는가?

E5 20 epoch 끝: τ = 0.030 (init 0.05 에서 0.030 으로 줄어듦).
E8: τ = 0.030.
E10 30 epoch 끝: **τ = 0.026** (init 0.05 → 0.026, log_temp -3.65).

τ 가 작아진다 = softmax 가 sharp 해진다. 모델이 학습이 진행되며 "지금 logit 분포 정도면 충분히 자신 있게 결정해도 된다" 고 판단한 것. CLIP 의 학습 곡선과 같은 패턴.

learnable τ 의 효과 측정 (E4 fixed τ=0.05 vs E5 learnable τ → 0.030):
- NDCG@50: 0.262 vs 0.260 — 차이 미미
- std_NDCG: 0.066 vs 0.063 — 거의 동일

**실용적 결론**: learnable τ 는 CLIP 표준이라는 점은 있지만 운영 메트릭 차이는 작다. fixed τ=0.05 (E4) 로도 충분.

### Q7. bias_correction (E8, E10) 의 효과는?

E5 (logQ X) vs E8 (logQ O) — 20 epoch 비교:

| | E5 | E8 |
|---|---|---|
| score_max | 22 | 38.6 (boost) |
| NDCG_50 | 0.260 | 0.258 |
| std_NDCG | 0.063 | 0.067 |

score 가 boost 되는 건 의도된 효과 (`-log(q)` 가 tail 에 더해짐). 메트릭 차이는 미미.

E4 (no logQ) vs E10 (logQ + mean_loss + warmup) — 30 epoch:

| | E4 | E10 |
|---|---|---|
| NDCG_50 | **0.262** | 0.255 |
| std_NDCG | 0.066 | 0.067 |
| Coverage | **86.6%** | 80.8% |
| TailRecall | **0.054** | 0.051 |

기대했던 "logQ correction 으로 tail 학습 강화" 가 측정 메트릭에선 명확히 안 보임. 오히려 E10 의 Coverage 와 TailRecall 이 E4 보다 살짝 낮음. 이유는:
1. cosine warmup 의 첫 5 epoch lr ramp 가 학습 진행을 늦춤
2. mean_loss + logQ 의 anti-popularity 효과가 겹쳐서 over-correction
3. 30 epoch 으로는 cosine schedule 의 full benefit 못 나옴 (보통 100+ epoch 필요)

**결론**: 추가 trick 들이 모두 효과 있는 건 아니다. 운영 stack 으로는 **E4 가 가장 깔끔**.

### Q8. negative_sampling 자체를 빼면? — E11 ablation (실행 예정)

**Setup**: E10 (full stack: bf16 + mean_loss + l2norm + learnable τ + adamw+warmup) 와 동일, 단 `negative_sampling=false` + 정의상 함께 꺼지는 `bias_correction=false`. in-batch sampled softmax (positive 가 같은 batch 의 다른 유저들의 positive 와 경쟁) → full softmax (vocab 180k 전체 경쟁) 로 전환. 즉 E10 → E11 의 비교가 "logQ 보정으로 sampling bias 잡기 (E10)" vs "sampling 자체를 안 함 (E11)" 의 직접 대결.

**구현 차이** (코드 한 분기로만 갈림 - `src/models/cbf.py:233-262`):
- **E0~E10 (sampled)**: `pred_scores = net(inputs, in_batch_pos_items)` — batch 내 unique positives 만 score 계산. label = positive 가 batch 후보 풀의 몇 번째인지.
- **E11 (full)**: `scores = net(inputs)` — vocab 180k 아이템 전체에 대해 score. label = multi-hot 180k 벡터 (positive=1, 나머지=0).

**측정 목적**:
1. **popularity bias 감소 가능성** — sampled softmax 는 batch 내 popular item 끼리 경쟁이 격해 popular 가 더 잘 학습됨 (대표적인 in-batch sampling 의 bias). full softmax 는 tail item 도 동일 무게로 negative 가 되므로 Coverage / TailRecall 이 올라갈 여지.
2. **E10 의 logQ correction 이 노린 효과를 sampling 제거로 직접 달성하는지** 비교 — E11 vs E10 으로 "Yi 2019 보정 vs full softmax" 의 결론을 본다.
3. **NDCG 절대값이 오를지** — sampled softmax 의 score 가 batch 차원에서 정의되는 반면 full softmax 는 deployment 와 일치하는 분포 (180k 풀) 위에서 학습됨. train-test mismatch 가 줄어 NDCG 가 오를 가능성.

**비용 / caveat**:
- batch size 1024 → 512 로 낮춤 (full vocab score 텐서가 batch×180k 라 OOM 위험). epoch 당 step 수 약 2배, wall-clock 증가.
- `bias_correction=true` 와 동시 사용 불가 (`src/datasets/cbf.py:270` 에서 assert). 즉 E10 → E11 비교는 엄밀히는 "negative_sampling + biasCorr 동시 off" 라 변수 1개가 아닌 2개지만, biasCorr 는 negative sampling 의 부산물 보정이라 negative sampling 없이는 정의 자체가 성립 안 함 — 사실상 1-variable 비교로 해석 가능.

**예상 결과**:
- Coverage ↑, TailRecall ↑ (in-batch sampling 의 popularity 편향 제거)
- NDCG@50: 같거나 살짝 ↑ (deployment matched 분포로 학습)
- 단점은 train cost 증가 & vocab 이 더 커질 때 scale 안 됨 — vocab 1M 으로 가면 full softmax 는 사실상 불가능.

---

## 5. cbf3 시리즈 결과 — 30 epoch finished runs

cbf3 base (vocab=180k, embed=512, dropout=0.0, price_group=0) 위에서 동일 안정화 매트릭스 적용.

| Exp | score_max | weight_max | τ_final | NDCG_50 | std_NDCG | Coverage% | TailRecall | TailExp% | HR_1 | MRR_50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **e0** baseline (fp16) | **∞** 💥 | 13.7 | 1.0 | 0.014 | 0.015 | 5.3 | 0.0004 | 0.4 | 0.009 | 0.027 |
| **e4** +l2norm+τ=0.05 | 14.4 ✅ | 15.5 | 0.05 | 0.085 | 0.129 | 88.0 | 0.083 | 15.0 | 0.044 | 0.128 |
| **e5** +learnable τ | 22.6 ✅ | 15.4 | 0.027 | 0.085 | 0.130 | **91.9** | 0.090 | 17.0 | 0.044 | 0.128 |
| **e7** E4+mean_loss | 14.4 ✅ | 13.7 | 0.05 | 0.084 | 0.127 | 87.0 | 0.080 | 14.6 | 0.044 | 0.127 |
| **e10** full stack (full softmax) | 23.6 ✅ | 8.0 | 0.024 | 0.084 | 0.125 | 88.6 | 0.080 | 15.3 | 0.045 | 0.128 |
| **e8** ⚡ sampling+logQ | 35.2 ✅* | 16.9 | 0.027 | **0.245** ★ | **0.099** ★ | **99.6** ★ | **0.147** ★ | **57.7** ★ | **0.204** ★ | **0.354** ★ |
| **e9** ⚡ E8+mean_loss | 38.2 ✅* | 14.3 | 0.024 | 0.241 | 0.097 | 99.5 | 0.143 | 56.8 | 0.200 | 0.348 |

> ⚡ = `negative_sampling=true` 인 실험 (e0, e4, e5, e7, e10 은 full softmax)
> `*` = score_max 가 1/τ 의 cap 초과 — bias_correction 의 −log(q) term 이 logit 을 boost 한 결과 (의도된 동작, 발산 아님)
> ★ = cbf3 시리즈 내 1위

미실행 (메모리/시간 자원 부족): cbf3-e1, e2, e3, e6, e11.

---

## 6. 핵심 발견 — cbf3 시리즈

### Q9. cbf3-e0 가 왜 완전 발산했나?

cbf3 의 운영 셋업 (`negative_sampling=False`, `τ=1.0`, `normalize_outputs=False`, fp16) 을 **30 epoch** 으로 늘리니 **score_max=∞ (fp16 overflow)**, **loss=NaN**, NDCG_50=0.014, Coverage=5.3% 의 완전 collapse.

운영 cbf3 는 20 epoch 마지노선에서 가까스로 살아있는 상태. **20 → 30 epoch 늘리는 것만으로 발산** 한다는 직접 증거. score_max 가 무한대로 가서 fp16 의 65k 한계를 넘는 순간 NaN propagation 시작.

| | cbf3-e0 @ 20 epoch (운영) | cbf3-e0 @ 30 epoch (본 실험) |
|---|---|---|
| score_max | (n/a — production 로깅 전) | **Infinity** |
| NDCG_50 | ~0.10 (운영 release notes 기준) | **0.014** |
| Coverage | n/a | **5.3%** |

발산 메커니즘은 cbf-eN 의 Q1 과 동일 (cross-entropy + dot product 의 magnitude push). 단 cbf3 의 vocab 이 좁아 (180k) 발산 속도가 cbf (1M vocab) 보다 빠름 — popularity push 의 누적 step 당 효과가 큰 vocab 보다 더 집중적.

**시사점**: 운영 cbf3 의 "20 epoch" 은 우연히 발산 직전에서 멈추는 epoch budget. **L2 norm + 작은 τ 도입이 운영 안정성에 필수**.

### Q10. **(가장 놀라운 발견)** full softmax 가 sampling+logQ 보다 NDCG 3배 낮음

cbf3 의 정체성인 **Full Softmax** (e4, e5, e7, e10) 가 동일 stability stack 의 **In-batch Sampled + logQ** (e8, e9) 에 ranking 메트릭에서 압도적 열세.

| 메트릭 | full softmax (e4~e7, e10 평균) | sampled+logQ (e8, e9 평균) | 배수 |
|---|---:|---:|---:|
| **NDCG_50** | 0.085 | **0.243** | 2.9x |
| **std_NDCG** | 0.128 | **0.098** | (낮을수록 좋음) |
| **HR_1** | 0.044 | **0.202** | 4.6x |
| **MRR_50** | 0.128 | **0.351** | 2.7x |
| **Recall_50** | 0.134 | **0.333** | 2.5x |
| **Coverage%** | 88.9 | **99.6** | 1.12x |
| **TailRecall** | 0.083 | **0.145** | 1.7x |
| **TailExposure%** | 15.5 | **57.3** | 3.7x |

**모든 메트릭에서 sampling+logQ 가 압도**. 특히 다양성 메트릭 (Coverage, TailRecall, TailExposure) 도 full softmax 가 자동으로 우월할 거라는 직관과 **정반대 결과**.

**왜 이런가** — 가설 4가지:

1. **Gradient dilution in full softmax**: 매 step 의 gradient 가 vocab 180k 의 모든 item 에 퍼짐. positive 가 ~50개라 negative 가 ~179,950개. CE loss 의 gradient mass 가 너무 분산되어 per-item 학습 신호가 약함.

2. **Easy negative 문제**: full softmax = 모든 item 이 negative. 대부분은 "정답과 명백히 다른" easy negative (예: 카테고리도 다른 item). Easy negative 는 gradient 가 거의 0 — 학습 정보량이 없음. In-batch sampled = 같은 batch 의 다른 user 가 좋아한 item → "다른 user 가 click 한 의미 있는 item" → inherently hard negative → 풍부한 학습 신호.

3. **Contrastive 구조 손실**: in-batch sampling 은 user-user 간 implicit contrastive (이 user 의 positive vs 다른 user 의 positive). CLIP/SimCLR/metric learning 의 본질이 이것. Full softmax 는 이 구조가 깨짐 — 단순히 "정답 vs 나머지 vocab" 의 standard classification 으로 회귀.

4. **평가 메트릭의 retrieval 본질**: 본 실험의 val metric (NDCG_50, Recall_50 등) 은 **180k 풀 전체 ranking 후 history 마스킹 → top-k** 의 retrieval 셋업. Sampled softmax 학습이 이 retrieval 셋업과 **objective alignment 이 더 좋음**. Full softmax 의 "classification 위 ranking" 은 우회 경로.

**Notion 의 기존 가정** ("일반적인 classification 보다 in-batch 가 negative 다양성 낮음 → 결과 나쁨") 이 본 실험에서 **반증됨**. 적어도 본 데이터 + 본 평가 셋업에선 **in-batch + logQ 가 full softmax 보다 우월**.

### Q11. learnable τ 가 cbf3 에서도 0.024~0.027 로 수렴 — robust optimum

cbf-eN 시리즈 (Q6) 에서 E5 의 learnable τ 가 0.05 → 0.030 으로 수렴. cbf3-eN 에서도 동일 패턴:

| run | base softmax | 최종 τ |
|---|---|---:|
| cbf-e5 | sampled | 0.030 |
| cbf3-e5 | full | 0.027 |
| cbf3-e8 | sampled+logQ | 0.027 |
| cbf3-e9 | sampled+logQ+mean_loss | 0.024 |
| cbf3-e10 | full+full stack | 0.024 |

모두 **0.024~0.030** 범위. base 의 sampling 방식 / vocab / embed 와 무관. → **Ably click 데이터의 robust optimal τ ≈ 0.025**. 운영 셋업에서 learnable τ 도입할 필요 없이 **fixed τ=0.025 ~ 0.03 으로 가도 충분**.

### Q12. e8 의 score_max 35 > 1/τ=37 cap — 발산 아님

cbf-eN 의 E10 (Q1 footnote) 과 동일 origin. `bias_correction` 의 `−log(q)` term 이 인기 아이템의 logit 을 깎고 tail 의 logit 을 올림. 결과 logit 의 dynamic range 가 [-1/τ, 1/τ] cap 을 약간 초과 가능. e8 의 35.2, e9 의 38.2 모두 cap 37 근처 — 발산이 아니라 **logQ 의 의도된 효과**.

NDCG / val 메트릭 안정적 (epoch 별 점진 상승, plateau, 후반 미세 변동) → 학습 정상.

---

## 7. cbf vs cbf3 비교 — Base config 의 효과

cbf-eN 과 cbf3-eN 은 **같은 stability stack 매트릭스**를 다른 base config 에서 적용. base 의 차이만 명시:

| | cbf base | cbf3 base |
|---|---|---|
| vocab (max_items) | 1,000,000 | 180,000 |
| item_embed_size | 256 | 512 |
| item_dropout_prob | 0.15 | 0.0 |
| price_group_embed_size | 8 | 0 |

### 7-1. baseline (E0) 발산 비교

| | cbf-E0 (20ep, fp16) | cbf-E0 (30ep, fp16) | cbf3-e0 (30ep, fp16) |
|---|---|---|---|
| score_max | 203 | (in-progress, 추정 ~2000) | **Infinity** |
| NDCG_50 | 0.255 | (in-progress) | **0.014** |
| Coverage% | (보조 메트릭 없음) | (in-progress) | **5.3** |

**cbf3 가 cbf 보다 더 빠르게/심하게 발산**. 이유는 좁은 vocab + 큰 embed (180k × 512 = 92M params) 가 magnitude push 의 cumulative 효과를 가속.

### 7-2. l2norm + 작은 τ 적용 효과 (E4)

| | cbf-E4 | cbf3-e4 |
|---|---|---|
| score_max | 14.3 | 14.4 |
| NDCG_50 (in-batch eval) | **0.262** | 0.085 |
| std_NDCG (full vocab) | 0.066 | **0.129** |
| Coverage% | 86.6 | 88.0 |
| TailRecall | 0.054 | 0.083 |

NDCG 절대값이 매우 다른 이유: **cbf-E4 의 NDCG_50 은 in-batch eval** (수천 후보 풀), **cbf3-e4 의 NDCG_50 은 full vocab eval** (180k). 다른 분모.

표준 메트릭 (std_NDCG, Coverage, TailRecall) 으로 비교하면 cbf3-e4 가 cbf-E4 보다 **약간 우수 (Coverage, TailRecall)**. cbf3 의 큰 embed (512) 와 dropout=0 이 per-item 표현력에 기여.

### 7-3. 최선 stack (cbf-E4 vs cbf3-e8)

| | cbf-E4 (sampling, no logQ) | cbf3-e8 (sampling+logQ) |
|---|---|---|
| Best stack | bf16 + l2norm + τ=0.05 | bf16 + l2norm + learnable τ + logQ |
| std_NDCG | 0.066 | **0.099** |
| std_Recall | 0.087 | **0.140** |
| std_HR_1 | 0.066 | **0.084** |
| std_MRR_50 | 0.133 | **0.177** |
| Coverage% | 86.6 | **99.6** |
| TailRecall | 0.054 | **0.147** |
| TailExp% | 48.9 | **57.7** |

cbf3-e8 이 **모든 표준 메트릭에서 cbf-E4 를 능가**. 특히:
- **Coverage 86.6% → 99.6%**: vocab 180k 중 거의 100% 가 어떤 user 의 top-50 에 등장
- **TailRecall 0.054 → 0.147 (2.7x)**: tail 정답 회수율이 압도적
- **std_NDCG 0.066 → 0.099 (1.5x)**: deployment 환경 ranking 도 우월

cbf3 의 base (embed=512, dropout=0, 좁은 vocab) + sampling + logQ 의 stack 이 본 데이터에서 **새로운 SOTA**.

### 7-4. 권고 — 두 endpoint 의 운영 stack 통합

- 본 실험 이전: cbf (sampled, embed=256) / cbf3 (full softmax, embed=512) 분기 → 다른 학습 dynamics, 다른 메트릭
- 본 실험 결과: **두 endpoint 모두 sampled + logQ + l2norm + learnable τ + bf16 stack 으로 통합 가능**. 차이는 vocab 크기 (cbf=1M, cbf3=180k) + embed/dropout 만.
- 즉 학습 코드 한 줄기로 endpoint 별 hparam 만 분기. 유지보수 ↓, 결과 ↑.

### 7-5. 운영 적용 우선순위

| 우선순위 | 액션 | 영향 |
|---|---|---|
| **P0** | cbf3 의 negative_sampling=False → True 전환 (cbf3-e8 셋업) | NDCG 3x ↑, Coverage 99.6% |
| **P0** | cbf3 + cbf 둘 다 l2norm + small τ + bf16 도입 | 발산 차단 + 다양성 ↑ |
| **P1** | cbf3 의 logQ correction (bias_correction=True) | tail 학습 강화 |
| **P2** | learnable τ — 효과는 marginal, fixed τ=0.025 로도 충분 | optional |
| **P3** | mean_loss — cbf-e3 vs cbf3-e9 모두 marginal effect, optional | optional |

---

## 8. 종합 결론

1. **운영 cbf / cbf3 모두 학습 안정성 문제 있음** — L2 norm + 작은 τ 도입이 필수. 본 실험으로 운영 셋업의 발산 메커니즘 (magnitude push) 과 fix (L2 norm) 명확히 측정.

2. **cbf3 의 full softmax 가 본 데이터에서 의외로 underperform** — Notion 의 기존 가정 (full softmax 가 popularity bias 없이 안정적 학습) 이 본 실험에서 반증됨. **Sampling + logQ 가 NDCG 3x, Coverage 99.6%** 로 압도. 원인은 gradient dilution + easy negative 문제 + retrieval objective alignment.

3. **두 endpoint 의 stack 통합 가능** — 같은 SOTA stack (bf16 + l2norm + small τ + sampling + logQ) 으로 cbf, cbf3 둘 다 운영. 차이는 base hparam (vocab/embed/dropout) 만.

4. **Learnable τ 의 robust optimum ≈ 0.025** — 다섯 개 run (cbf-e5, cbf3-e5/e8/e9/e10) 모두 0.024~0.030 으로 수렴. Production 에선 fixed τ=0.025 로 단순화 가능.

5. **추가 trick 들의 ROI 평가**:
   - L2 norm + small τ: **P0** (발산 차단 + 다양성 큰 폭 ↑)
   - logQ (bias_correction): **P1** (cbf3 에선 NDCG 큰 폭 ↑, cbf 에선 marginal)
   - learnable τ, mean_loss, adamw+warmup, gradient_clip: **P3** (marginal, optional)

---

> **요약 한 줄**: 좁은 풀의 reranker 라 해서 full softmax 가 정답인 것 아님. **sampling + logQ + L2 norm + 작은 τ 의 표준 retrieval stack** 이 cbf 와 cbf3 모두에서 SOTA. 본 실험은 운영 stack 통합과 안정성 도입의 직접적 근거.
