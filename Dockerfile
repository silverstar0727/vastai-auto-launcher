FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# AWS CLI 설치
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Vast.ai CLI 설치
RUN uv pip install --system vastai

# 작업 디렉토리 설정
WORKDIR /workspace

# 의존성 파일 복사 및 설치
COPY pyproject.toml .
RUN uv pip install --system .

# 소스 코드 복사
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN chmod +x scripts/entrypoint.sh

# 환경변수 (실행 시 override 가능)
ENV WANDB_API_KEY=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION="ap-northeast-2"
ENV S3_DATA_PATH=""
ENV CONTAINER_ID=""
ENV VAST_API_KEY=""

ENTRYPOINT ["scripts/entrypoint.sh"]
