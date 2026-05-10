# reco-lightning 학습 가이드

이 저장소는 inhouse 추천 모델들을 PyTorch Lightning + Lightning CLI 기반으로
포팅한 프로젝트다. 한 줄 요약: **`configs/<model>.yaml` 한 파일로 모델·데이터·트레이너 설정을 묶고
`python src/main.py fit -c configs/<model>.yaml` 한 줄로 학습**.

학습은 두 가지 환경에서 동일 코드로 돌릴 수 있다:
1. **로컬 (uv)** — 개발/디버깅
2. **Runpod (docker image)** — 본격 학습. silverstar456/ably-reco-lightning 이미지가 git clone 으로 코드를 받아 실행

---

## 0. 모델 / 브랜치 매트릭스

| 브랜치 | yaml | 차이점 (vs `cbf-reprod` 운영 baseline) | 비고 |
|---|---|---|---|
| `main` | `cbf.yaml`, `cbf-reprod.yaml` 등 | — (cbf-reprod = 운영 align) | 새 실험 시작점 |
| `exp-e0-baseline` | `cbf-e0.yaml` | 없음 (재현용) | overflow 발생 시점 측정 baseline |
| `exp-e1-bf16` | `cbf-e1.yaml` | `precision: bf16-mixed` | fp16 overflow 회피 |
| `exp-e2-clip` | `cbf-e2.yaml` | bf16 + `gradient_clip_val: 1.0` | weight 폭주 억제 |
| `exp-e3-meanloss` | `cbf-e3.yaml` | bf16 + `mean_loss: true` | user 별 loss 균등화 |
| `exp-e4-l2norm-fixed-tau` | `cbf-e4.yaml` | bf16 + `normalize_outputs: true` + `softmax_temperature: 0.05` | l2norm 단독 효과 |
| `exp-e5-l2norm-learnable-tau` | `cbf-e5.yaml` | bf16 + l2norm + 학습 가능 τ | 이민호님 권장. AB 후보. **코드 변경 포함** |

> 모든 브랜치에 **score_min/score_max/weight_max** 진단 로깅 포함 (운영 cbf3 의 'Score 발산 추이' 와 같은 포맷, wandb).
> E5 만 `train/temperature` `train/log_temperature` 추가.

`MODEL` ↔ `REPO_REF` 매핑은 항상 짝 (E1 yaml 을 E0 브랜치에서 못 찾음):

| MODEL env | REPO_REF env |
|---|---|
| `cbf-e0` | `exp-e0-baseline` |
| `cbf-e1` | `exp-e1-bf16` |
| `cbf-e2` | `exp-e2-clip` |
| `cbf-e3` | `exp-e3-meanloss` |
| `cbf-e4` | `exp-e4-l2norm-fixed-tau` |
| `cbf-e5` | `exp-e5-l2norm-learnable-tau` |

---

## 1. 데이터 소스

운영 학습은 매일 **00:05 KST 에 reco_raw_parquet**, **01:45 KST 에 interaction CSV** 가 생성되고
다음 날 새벽에 학습이 시작됨. 즉 N일 새벽 학습 = `dt=(N-1)`.

| 데이터 | S3 경로 | 버킷 | runpod 권한 |
|---|---|---|---|
| Interaction CSV | `s3://ably-analytics-data/personalize/inhouse/{date}/` | ably-analytics-data | ✅ |
| Pretrained text | `s3://ably-analytics-data/models/pretrained_text/` | ably-analytics-data | ✅ |
| goods.parquet | `s3://ably-datalake/reco/db/reco_raw_parquet/goods/dt={date}/` | **ably-datalake** | ⚠️ DevOps 요청 |
| order.parquet | `.../order/dt={date}/` | ably-datalake | ⚠️ |
| category.parquet | `.../category/dt={date}/` | ably-datalake | ⚠️ |
| standard_category.parquet | `.../standard_category/dt={date}/` | ably-datalake | ⚠️ |
| member.parquet | `.../member/dt={date}/` | ably-datalake | ⚠️ |
| goods_attribute_values.parquet | `.../goods_attribute_values/dt={date}/` | ably-datalake | ⚠️ (use_attr=True 만) |
| goods_attribute_field_values.parquet | `.../goods_attribute_field_values/dt={date}/` | ably-datalake | ⚠️ (use_attr=True 만) |

**runpod IAM user (`runpod@prod`) 에 `ably-datalake/reco/db/reco_raw_parquet/*` GetObject + ListBucket 권한 추가 필요** — 없으면 cbf-reprod / cbf-e* 모두 goods.parquet 다운에서 403.
일자별 다운 사이즈: ~5GB.

---

## 2. 로컬 학습 (uv)

### 2-1. 사전 준비
```bash
cd /home/jeongmindo/projects/ably/reco-lightning
uv sync                                             # 의존성 (한 번)
aws sso login --profile MLEngineer-145481888492     # 12h 토큰
export AWS_PROFILE=MLEngineer-145481888492
uv run python -m wandb login --relogin              # wandb 직접
```

### 2-2. 데이터 다운 (호스트)
```bash
DATE=20260506
DATA=/home/jeongmindo/projects/ably/data
mkdir -p $DATA/bert/interaction $DATA/sql $DATA/pretrained_text

aws s3 sync s3://ably-analytics-data/personalize/inhouse/$DATE/ \
    $DATA/bert/interaction/ --exclude "_SUCCESS"
aws s3 sync s3://ably-analytics-data/models/pretrained_text/ \
    $DATA/pretrained_text/

RECO=s3://ably-datalake/reco/db/reco_raw_parquet
for tbl in goods order category standard_category member \
           goods_attribute_values goods_attribute_field_values; do
    aws s3 cp $RECO/$tbl/dt=$DATE/$tbl.parquet $DATA/sql/$tbl.parquet
done
```

### 2-3. 학습
```bash
uv run python src/main.py fit -c configs/cbf-reprod.yaml          # 운영 align baseline
uv run python src/main.py fit -c configs/cbf-e1.yaml              # E1 (bf16) — 단 E1 yaml 은 exp-e1-bf16 브랜치에 있으니 git checkout 필요
```

> 로컬에서 실험 브랜치를 돌리려면 `git checkout exp-e1-bf16` 후 `configs/cbf-e1.yaml` 사용.

---

## 3. Runpod 학습

이미지: `silverstar456/ably-reco-lightning:latest` — **빌드 시점엔 의존성과 부트스트랩만 baked**, 코드는 매 pod 시작 시 git clone.

### 3-1. 필수 환경변수

| Key | Value | 설명 |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `{{ RUNPOD_SECRET_aws_access_key_id }}` | 사내 runpod Secret |
| `AWS_SECRET_ACCESS_KEY` | `{{ RUNPOD_SECRET_aws_secret_access_key }}` | 사내 runpod Secret |
| `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_github_token }}` | private repo clone (사용자 본인 발급 PAT) |
| `WANDB_API_KEY` | `wandb_v1_...` 직접 또는 secret | |
| `MODEL` | `cbf-e0` ~ `cbf-e5` 또는 `cbf-reprod` | 위 0번 매트릭스 참고 |
| `DATE` | `YYYYMMDD` (예: `20260506`) | dt 파티션 |
| `REPO_REF` | `exp-eN-...` (또는 `main`) | MODEL 과 짝 |

### 3-2. 선택 환경변수 (default 있음)

| Key | Default | 변경 케이스 |
|---|---|---|
| `WANDB_PROJECT` | `cbf-exp` | 프로젝트 분리 시 |
| `REPO_URL` | `https://github.com/ml-jeongmin-do/runpod-launcher.git` | 다른 fork |
| `AWS_DEFAULT_REGION` | `ap-northeast-2` | |
| `DATA_ROOT` | `/data` | |
| `SKIP_DOWNLOAD` | `0` | `1` 이면 다운 안 함 (volume 으로 데이터 영속 시) |
| `SKIP_UV_SYNC` | `0` | `1` 이면 uv sync 건너뜀 (deps 변경 없을 때 빠름) |
| `KEEP_LOGS` | `0` | `1` 이면 `/workspace/logs` 보존 (resume 시도) |
| `KEEP_MODEL_PATH` | `0` | `1` 이면 `$DATA_ROOT/output/$MODEL` 보존 (preprocessed cache) |
| `GIT_FETCH_RETRIES` | `3` | |

### 3-3. ❌ 넣지 말 것

| Key | 이유 |
|---|---|
| `AWS_PROFILE` | access key 와 충돌 |
| `AWS_SESSION_TOKEN` | 장기 IAM key 에는 없음 |

### 3-4. Pod 스펙

| 항목 | 값 |
|---|---|
| GPU | RTX PRO 6000 / RTX 6000 Ada / A6000 1장 |
| Container disk | **200 GB** (이미지 ~35GB + 데이터 ~25GB + ckpt ~5GB + 여유) |
| Volume | 0 GB (영속 필요 없음) |
| Container Start Command | (비워둠) |
| Expose Ports | (없음) |

### 3-5. Pod 시작 시 자동 동작 (entrypoint)

1. **AWS 인증 확인** (`aws sts get-caller-identity`)
2. **이전 run 잔여 정리**:
   - `/workspace/logs` 삭제 (`KEEP_LOGS=1` 로 비활성화)
   - `$DATA_ROOT/output/$MODEL` 삭제 (`KEEP_MODEL_PATH=1` 로 비활성화)
3. **데이터 다운로드** (S3 → `/data`)
4. **wandb login**
5. **학습 시작** — yaml 의 호스트 절대경로는 entrypoint 가 자동으로 `/data/...` 로 override

### 3-6. 실행 흐름

```
┌─ pod 시작 ─────────────────────────────────────┐
│ image baked: /usr/local/bin/runpod-bootstrap.sh │
│   ├─ git clone (REPO_URL @ REPO_REF)            │
│   ├─ uv sync --frozen                            │
│   └─ exec /workspace/scripts/runpod_entrypoint.sh
│         ├─ AWS auth check                        │
│         ├─ /workspace/logs, output/$MODEL 정리   │
│         ├─ S3 다운 (interaction + parquet 7종)   │
│         ├─ wandb login                           │
│         └─ python src/main.py fit -c configs/$MODEL.yaml
│             + auto path overrides /data/...      │
└──────────────────────────────────────────────────┘
```

---

## 4. 진단 메트릭 (모든 실험 브랜치 공통)

운영 cbf3 의 'Score 발산 추이' 표와 같은 포맷:

| wandb metric | 의미 | 어디서 |
|---|---|---|
| `train/loss` | epoch 평균 loss | LossAccumulator |
| `train/score_min` | post-temperature logit 의 min (epoch 내 batch reduce) | training_step |
| `train/score_max` | 위 max | training_step |
| `train/weight_max` | trainable param 의 max abs (epoch 끝) | on_train_epoch_end |
| `train/temperature` (E5) | learnable τ 의 현재 값 | on_train_epoch_end |
| `train/log_temperature` (E5) | raw nn.Parameter 값 | on_train_epoch_end |
| `val/NDCG_50`, `val/Recall_50` | 검증 지표 (기존) | validation_step |

**score range** = `score_max - score_min` 은 wandb derived metric 으로 만들면 운영 표 그대로 재현. 발산 / NaN 시점 즉시 가시화.

---

## 5. 이미지 빌드/푸시

이미지 = base + uv + lock + bootstrap. **bootstrap.sh, Dockerfile.runpod 변경은 재빌드 필요**.
**그 외 (entrypoint, download, configs, src/) 는 git push 만으로 반영** (런타임 clone).

```bash
DOCKERHUB_USER=silverstar456 bash scripts/runpod_build_push.sh
```

빌드 캐시:
- pyproject.toml / uv.lock 안 바뀌면 uv sync layer 재사용 → 5분 이내
- bootstrap.sh 만 바뀌면 수십 초

### 재빌드가 필요한 변경
- `Dockerfile.runpod`
- `pyproject.toml` / `uv.lock`
- `scripts/runpod_bootstrap.sh`

### git push 만으로 반영되는 변경
- `scripts/runpod_entrypoint.sh`
- `scripts/runpod_download.sh`
- `configs/*.yaml`
- `src/**`

---

## 6. 흔한 문제

| 증상 | 원인 / 해결 |
|---|---|
| 다운로드에서 `403 Forbidden` (HeadObject) | runpod IAM 에 `ably-datalake` 권한 없음. DevOps 에 요청 (위 1번 표) |
| `[download] WARN: cbf-e? 에 대한 다운로드 정의 없음` | `runpod_download.sh` 의 case 패턴 누락 — `cbf-e[0-9]*` 추가 필요 (현 main 에 반영됨) |
| `ValueError: No objects to concatenate` | 위 다운 누락의 후속. 다운 fix 후 재실행 |
| `FileNotFoundError: 'logs/.../v1/fit/config.yaml'` | 이전 실패 run 의 잔여. entrypoint 의 cleanup 이 처리 (`KEEP_LOGS=0` 기본) |
| `Tini is not running as PID 1` 경고 | runpod 자체 init 이 PID 1. 무해, 무시 |
| 학습이 매우 느림 | num_workers=0 기본. yaml 에 `num_workers: 8` 또는 cmdline `--data.init_args.num_workers 8` |
| `An error occurred (ExpiredToken)` | SSO 만료 (호스트 인증 케이스에만 해당). runpod 의 IAM key 는 만료 없음 |
| 다른 모델로 갈아끼웠는데 옛 데이터 사용 | `KEEP_MODEL_PATH=0` (default) 면 자동 정리됨. `1` 이었다면 `0` 으로 |
| ckpt 가 5GB+ 로 큼 | 정상. 1.3B params (frozen text + 1007차원 attr_vector lookup table 포함) |

---

## 7. 빠른 시작 — Runpod

```
□ DevOps 에 ably-datalake 권한 요청 (필수)
□ runpod Secrets: aws_access_key_id, aws_secret_access_key, github_token (이름 등록)
□ Template 환경변수 설정 (위 3-1, 3-2)
□ Image: silverstar456/ably-reco-lightning:latest (또는 digest 명시)
□ Container disk 200GB
□ Deploy → Logs 탭에서 [bootstrap] / [entrypoint] / [download] 진행 확인
□ wandb https://wandb.ai/<account>/cbf-exp/runs/ 에서 score_min/score_max/loss 곡선 확인
```

실험 한 사이클:
```
MODEL=cbf-e0 REPO_REF=exp-e0-baseline   → baseline overflow 시점 측정
MODEL=cbf-e1 REPO_REF=exp-e1-bf16        → bf16 만으로 막아지는지
... 차례로 ...
MODEL=cbf-e5 REPO_REF=exp-e5-l2norm-learnable-tau → 최종 후보
```

각 실험은 wandb 별도 run 으로 기록 (yaml 의 `name` 이 다름). 6개 곡선 한 dashboard 에 겹쳐 비교.
