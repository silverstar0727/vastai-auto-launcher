FROM vastai/pytorch:2.9.1-cuda-13.0.2-py312-24.04

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# AWS CLI 설치
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# 작업 디렉토리 설정
WORKDIR /workspace

# conda 환경 PATH 설정
ENV PATH="/venv/main/bin:$PATH"

# 의존성 파일 복사 및 설치
COPY pyproject.toml .
RUN uv pip install --python /venv/main/bin/python --no-cache .

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
