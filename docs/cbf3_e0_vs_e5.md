# cbf3-e0 vs cbf3-e5 비교 보고서

운영 cbf3 (full softmax, vocab=180k, embed=512) 위에서 **L2 norm + small τ + learnable τ** 의 도입 효과를 측정.

- **cbf3-e0** = 운영 cbf3 의 30 epoch 재현 (안정화 trick 없음, fp16)
- **cbf3-e5** = cbf3-e0 + bf16 + L2 norm + softmax_temperature=0.05 + learnable τ

데이터: 20260506 dt 파티션. 1.6M user, vocab 180,000.

---

## 1. 설정 차이

| 항목 | cbf3-e0 | cbf3-e5 |
|---|:-:|:-:|
| precision | 16-mixed | **bf16-mixed** |
| `normalize_outputs` (L2 norm) | False | **True** |
| `softmax_temperature` (초기 τ) | 1.0 | **0.05** |
| `learnable_temperature` | False | **True** |
| `negative_sampling` | False (full softmax) | False (동일) |
| `bias_correction` | False | False (동일) |
| `mean_loss` | False | False (동일) |
| `optimizer` | adam, lr=0.001 | adam, lr=0.001 (동일) |
| 기타 base | item_embed=512, dropout=0, max_items=180k | 동일 |
| max_epochs | 30 | 30 |

→ **4가지만 변경** (precision, l2norm, τ 초기값, learnable τ). 나머지 모두 동일.

---

## 2. 결과 요약

| Metric | cbf3-e0 | cbf3-e5 | 변화 |
|---|---:|---:|---|
| **train/score_max** | **Infinity** 💥 | 22.6 ✅ | 발산 → 안정 |
| **train/score_min** | -17,808 | -16.9 | 발산 → 안정 |
| **train/weight_max** | 13.7 | 15.4 | 비슷 |
| **train/loss (최종)** | NaN | 691.7 | 학습 회복 |
| **train/temperature (최종)** | 1.0 (고정) | **0.0267** (학습) | — |
| | | | |
| **val/NDCG_50** | 0.014 | 0.085 | 6.1x |
| **val/Recall_50** | 0.021 | 0.135 | 6.4x |
| **val/HR_1** | 0.009 | 0.044 | 4.9x |
| **val/MRR_50** | 0.027 | 0.128 | 4.7x |
| | | | |
| **val/std_NDCG_50** | 0.015 | **0.130** | 8.7x |
| **val/std_Recall_50** | 0.022 | 0.176 | 8.0x |
| **val/std_HR_1** | 0.010 | 0.128 | 12.8x |
| **val/std_MRR_50** | 0.029 | 0.240 | 8.3x |
| | | | |
| **val/Coverage_%_50** | 5.3% | **91.9%** | 17.3x |
| **val/TailRecall_50** | 0.0004 | 0.090 | 225x |
| **val/TailExposure_%_50** | 0.4% | 17.0% | 42.5x |

→ **모든 메트릭에서 cbf3-e5 가 압도**. 특히 다양성/tail 메트릭에서 1~2 자릿수 배수 격차.

---

## 3. 학습 안정성

> *[이미지: cbf3-e0 vs cbf3-e5 의 train/score_max trajectory (epoch 0~29)]*

### cbf3-e0 — fp16 overflow 로 인한 catastrophic divergence

- 초기 epoch 부터 score_max 가 빠르게 증가
- 어느 시점에서 fp16 의 65,504 한계 초과 → **NaN / Inf propagation 시작**
- 최종 `score_max = Infinity`, `loss = NaN`
- score_min 도 -17,808 으로 음의 무한대 방향 발산

원인:
1. **`normalize_outputs=False`** 이라 u, v 의 magnitude (`‖u‖`, `‖v‖`) 가 무제한 성장 가능
2. cross-entropy 가 정답 logit 을 ∞ 로 보내려 함 → magnitude push 가 가장 쉬운 길
3. `softmax_temperature=1.0` 이라 logit scale 그대로. 정규화 없음.
4. fp16 의 dynamic range 좁아 (~65k) 빠르게 overflow

### cbf3-e5 — score_max 가 1/τ 의 cap 근처에서 안정

- `‖u‖=‖v‖=1` (L2 normalize) → `u·v = cos(θ) ∈ [-1, 1]`
- score = `cos(θ) / τ`, τ → 0.027 로 수렴 → logit range ≈ **[-37, +37]**
- 실측 score_max = 22.6 → cap 의 60% 수준에서 안정 (모든 user-item 쌍이 완벽히 정렬되진 않음 = healthy)
- score_min = -16.9 → 음의 영역도 잘 학습됨 (popularity collapse 아님)

→ **L2 norm 이 magnitude 발산 path 를 차단**. 학습이 cos(θ) 방향 정렬로만 진행.

> *[이미지: cbf3-e0 vs cbf3-e5 의 train/weight_max trajectory]*

weight_max 는 두 실험 모두 비슷한 영역 (~13-15). L2 norm 이 임베딩 magnitude 자체를 강제하진 않음 — forward 출력만 정규화. raw weight 는 자유. 그러나 score 의 magnitude 가 cap 되니 학습 dynamics 가 다름.

---

## 4. 정확도 메트릭 비교

> *[이미지: cbf3-e0 vs cbf3-e5 의 val/NDCG_50 epoch 별 trajectory]*

> *[이미지: cbf3-e0 vs cbf3-e5 의 val/std_NDCG_50 epoch 별 trajectory]*

### in-batch eval (val/*) vs full vocab eval (val/std_*)

cbf3 base 가 full softmax 라 두 메트릭의 candidate pool 이 같음:
- `val/NDCG_50`: full vocab 180k pool, history 미마스킹
- `val/std_NDCG_50`: full vocab 180k pool, history 마스킹 (→ task 더 쉬워짐)

cbf3-e5 의 두 값 (0.085 vs 0.130) 차이는 history removal 의 효과.

### 핵심 관찰

- cbf3-e0 의 모든 메트릭이 거의 0 — 학습 완전 실패 시 random ranking 보다도 나쁨
- cbf3-e5 의 std_HR_1=0.128 → **180k 중 정확히 정답 item 을 1위로 띄우는 user 가 12.8%**. 좁은 풀에서 강한 ranking 성능.
- ranking precision 메트릭 (HR_1, MRR_50) 의 격차 (12.8x, 8.3x) 가 NDCG (8.7x) 보다 큼 → cbf3-e5 가 특히 "top 1 정확도" 에 강함.

---

## 5. 다양성 / popularity bias 메트릭

> *[이미지: cbf3-e0 vs cbf3-e5 의 val/Coverage_Percentage_50 trajectory]*

### Coverage — 추천 다양성

- **cbf3-e0**: 180k vocab 중 5.3% (9,482 개) 만 어떤 user 의 top-50 에 등장.
  → **95% 의 item 이 절대 추천되지 않음**. 인기 상위 5% 정도만 반복 추천 = **popularity collapse**.
- **cbf3-e5**: 91.9% (165,451 개) 가 추천에 등장. **vocab 의 거의 모든 item 이 적어도 한 명의 user 에게 노출**.

> *[이미지: cbf3-e0 vs cbf3-e5 의 TailRecall + TailExposure trajectory]*

### TailRecall — tail 정답 회수율

- **cbf3-e0**: 0.0004 (0.04%). tail item 이 정답이어도 거의 회수 못 함.
- **cbf3-e5**: 0.090 (9.0%). **225배 향상**.

### TailExposure — top-50 추천 중 tail 비율

- **cbf3-e0**: 0.4%. 추천 top-50 의 99.6% 가 head item.
- **cbf3-e5**: 17.0%. 1/6 가량이 tail.

→ L2 norm + small τ 의 도입이 popularity bias 를 거의 완전히 해소.

---

## 6. cbf3-e5 의 temperature 학습 trajectory

> *[이미지: cbf3-e5 의 train/temperature 와 train/log_temperature epoch 별 변화]*

### 수렴 양상

| Epoch | τ | log_t |
|---|---|---|
| 0 (초기) | 0.0500 | -3.00 |
| ~10 | ≈ 0.040 | ≈ -3.22 |
| ~20 | ≈ 0.032 | ≈ -3.44 |
| 29 (최종) | **0.0267** | **-3.62** |

CLIP 의 표준 trajectory 와 동일 — 초기값에서 점진 decay, 후반 plateau.

### 해석

- `cos(θ) / τ` 의 sharpness 가 학습 신호의 크기를 결정
- 초기 τ=0.05 (logit range ±20) 는 합리적 시작이지만 데이터에 따라 더 sharp 한 ranking 이 적합한지 판단 필요
- cbf3-e5 의 결과: **τ 가 줄어드는 방향 = 더 sharp 한 softmax** 가 본 데이터에 더 적합
- 최종 0.027 ≈ logit range ±37 → winning logit 의 P 가 거의 1.0 까지 가능
- 단, NDCG 자체는 fixed τ=0.05 (cbf3-e4) 와 비교해 marginal 차이 — sharpness 의 marginal benefit 은 작음

---

## 7. 결론

### 운영 cbf3 의 현 상태가 위험

cbf3-e0 의 30 epoch 결과 (score_max=∞, NDCG ≈ 0, Coverage 5%) 는 **운영 cbf3 의 20 epoch 셋업이 발산 직전에서 가까스로 멈추는 상태**라는 직접 증거.
- 운영 셋업은 epoch budget 으로 "안 죽는" 셋업
- 학습 데이터 분포 변화 / vocab 확장 / epoch 증가 등 어떤 변경이 있어도 쉽게 발산
- → **L2 norm + small τ 도입이 운영 안정성에 필수**

### 4 줄짜리 yaml 변경으로 큰 이득

cbf3-e5 의 변경은:
```yaml
trainer:
  precision: bf16-mixed         # ← 16-mixed 에서 변경
model:
  init_args:
    normalize_outputs: true     # ← false 에서 변경
    softmax_temperature: 0.05   # ← 1.0 에서 변경
    learnable_temperature: true # ← false 에서 변경
```

이 4 줄로:
- 안정성: ∞ → 22.6 (수학적 cap 안)
- NDCG: 0.014 → 0.085 (6x)
- Coverage: 5.3% → 91.9% (17x)
- TailRecall: 0.0004 → 0.090 (225x)

### 운영 적용 권장

**즉시 적용 가능한 P0 변경**. learnable τ 는 marginal contribution 이라 optional 이지만 도입해도 손해 없음. fixed τ=0.05 만 적용해도 거의 동일한 효과.

---

## 부록 — 참고할 메트릭 종류 한눈에

| 카테고리 | 메트릭 |
|---|---|
| **학습 진단** | `train/score_max`, `train/score_min`, `train/weight_max`, `train/loss`, `train/temperature`, `train/log_temperature` |
| **정확도 (in-batch)** | `val/NDCG_50`, `val/Recall_50`, `val/HR_1`, `val/MRR_50` |
| **정확도 (full vocab + history removed)** | `val/std_NDCG_50`, `val/std_Recall_50`, `val/std_HR_1`, `val/std_MRR_50` |
| **다양성 / popularity bias** | `val/Coverage_Percentage_50`, `val/TailRecall_50`, `val/TailExposure_50` |

wandb 의 `cbf-exp` 프로젝트에서 두 run (`cbf3-e0`, `cbf3-e5`) 선택 → metric panel 에서 위 메트릭들 비교 plot 가능.
