#!/bin/bash
# Pod 시작 시 호출되는 부트스트랩.
# 1) GitHub 에서 코드 fetch+checkout → /workspace
# 2) (선택) uv sync --frozen — uv.lock 변경분 따라잡기
# 3) /workspace/scripts/runpod_entrypoint.sh 로 exec
#
# 환경변수:
#   REPO_URL           https://github.com/<owner>/<repo>.git  (필수, default 있음)
#   REPO_REF           branch | tag | commit SHA              (default: main)
#   GITHUB_TOKEN       private repo 시 필수 (runpod Secret 권장)
#   SKIP_UV_SYNC       1 이면 런타임 uv sync 건너뜀
#   GIT_FETCH_RETRIES  default 3
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ml-jeongmin-do/runpod-launcher.git}"
REPO_REF="${REPO_REF:-main}"
GIT_FETCH_RETRIES="${GIT_FETCH_RETRIES:-3}"

log()  { echo "[bootstrap] $*"; }
fail() { echo "[bootstrap] ERROR: $*" >&2; exit 2; }

# ---- 1) AUTH URL 구성 (token 은 ps 에 잠깐 노출되지만 single-user pod 내라 허용) ----
if [ -n "${GITHUB_TOKEN:-}" ]; then
    AUTH_URL="${REPO_URL/https:\/\//https://oauth2:${GITHUB_TOKEN}@}"
else
    AUTH_URL="$REPO_URL"
fi

# ---- 2) /workspace 에 fetch+checkout (init 패턴, dirty workspace 도 OK) ----
mkdir -p /workspace
cd /workspace

if [ ! -d ".git" ]; then
    log "git init (empty 또는 dirty workspace 모두 OK)"
    git init -q
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$AUTH_URL"

# fetch 실패 시 재시도 (일시 network 이슈 대비)
for i in $(seq 1 "$GIT_FETCH_RETRIES"); do
    if git fetch --depth 1 origin "$REPO_REF" 2>/tmp/git-fetch.err; then
        break
    fi
    if [ "$i" = "$GIT_FETCH_RETRIES" ]; then
        cat /tmp/git-fetch.err >&2
        # token 미정의 + private repo 의심 메시지
        if ! grep -q oauth2 <<< "$AUTH_URL"; then
            log "  hint: private repo 면 GITHUB_TOKEN 환경변수(또는 runpod Secret) 확인"
        fi
        fail "git fetch 실패: $REPO_URL ($REPO_REF) — ${GIT_FETCH_RETRIES}회 재시도 후 실패"
    fi
    log "git fetch 실패 재시도 $i/$GIT_FETCH_RETRIES"
    sleep 2
done

git reset --hard FETCH_HEAD
git remote set-url origin "$REPO_URL"   # 토큰 git config 에 안 남도록 복원
log "현재 commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s 2>/dev/null | head -c 80))"

# ---- 2.5) 이전 run 의 잔여 산출물 정리 (runpod container disk 가 stop/start 사이 영속이라 필요) ----
# /workspace/logs : lightning + wandb 로컬 dir. 남아있으면 _check_resume 가 죽은 run 의
#                   config.yaml 찾다 FileNotFoundError 로 학습 시작 자체를 막음.
# $DATA_ROOT/output/$MODEL : preprocessed cache, encoder JSON 등. 같은 데이터면 재생성해도
#                   결과 동일하지만 cache miss 로 setup 시간 늘어남 (보존이 더 나을 수도).
# 환경변수 KEEP_LOGS=1 / KEEP_MODEL_PATH=1 로 각각 보존 가능.
if [ "${KEEP_LOGS:-0}" != "1" ] && [ -d /workspace/logs ]; then
    log "이전 logs/ 정리 (KEEP_LOGS=1 로 비활성화 가능)"
    rm -rf /workspace/logs
fi
if [ "${KEEP_MODEL_PATH:-0}" != "1" ] && [ -n "${MODEL:-}" ] && [ -d "${DATA_ROOT:-/data}/output/$MODEL" ]; then
    log "이전 ${DATA_ROOT:-/data}/output/$MODEL 정리 (KEEP_MODEL_PATH=1 로 비활성화 가능)"
    rm -rf "${DATA_ROOT:-/data}/output/$MODEL"
fi

# ---- 3) (선택) uv sync ----
if [ "${SKIP_UV_SYNC:-0}" = "1" ]; then
    log "SKIP_UV_SYNC=1 → uv sync 건너뜀"
elif [ -f pyproject.toml ] && [ -f uv.lock ]; then
    # 빌드 시점 lock(/opt/project) 과 다르면 실제로 해소. 같으면 instant.
    log "uv sync --frozen --no-dev"
    if ! uv sync --frozen --no-dev 2>/tmp/uv-sync.err; then
        log "WARNING: uv sync 실패. 자세한 로그:"
        cat /tmp/uv-sync.err >&2
        log "→ 빌드 시점에 박힌 /opt/venv 그대로 사용해서 진행 (의존성 변경 있으면 이미지 rebuild 필요)"
    fi
else
    log "pyproject.toml/uv.lock 없음 → uv sync 생략"
fi

# ---- 4) entrypoint 로 exec ----
ENTRYPOINT_SCRIPT="/workspace/scripts/runpod_entrypoint.sh"
[ -f "$ENTRYPOINT_SCRIPT" ] || fail "$ENTRYPOINT_SCRIPT 없음. repo 구조 확인."
chmod +x "$ENTRYPOINT_SCRIPT"

log "→ $ENTRYPOINT_SCRIPT 실행"
exec "$ENTRYPOINT_SCRIPT" "$@"
