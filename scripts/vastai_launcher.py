#!/usr/bin/env python3
"""
Vast.ai 자동 인스턴스 선택 및 실행 스크립트
- 조건에 맞는 최적의 인스턴스를 자동 선택
- 지정된 Docker 이미지로 실행
- GitHub Actions에서 자동 호출 가능
"""

import argparse
import requests
import json
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# 설정
# ============================================================

@dataclass
class Config:
    # Vast.ai API 키
    api_key: str = ""

    # Docker 이미지 설정
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"

    # 인스턴스 요구사항
    max_price_per_hour: float = 2.0
    min_reliability: float = 0.9
    min_disk_space: float = 50.0
    min_inet_down: float = 100.0
    min_inet_up: float = 100.0
    num_gpus: int = 1

    # 선호 GPU
    preferred_gpus: tuple = ("RTX 4090",)

    # 실행 설정
    disk_space: float = 50.0

    # 환경변수 (Docker 컨테이너에 전달)
    env_vars: dict = field(default_factory=dict)

    # 학습 설정
    config_file: str = "configs/config.yaml"
    extra_args: str = ""


# ============================================================
# API 클라이언트
# ============================================================

class VastClient:
    BASE_URL = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Accept": "application/json"}

    def search_offers(self) -> list:
        """사용 가능한 인스턴스 목록 조회"""
        url = f"{self.BASE_URL}/bundles/"
        params = {"api_key": self.api_key}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json().get("offers", [])

    def create_instance(self, offer_id: int, image: str, disk: float,
                        onstart: str = "", env: dict = None) -> dict:
        """인스턴스 생성 및 실행"""
        url = f"{self.BASE_URL}/asks/{offer_id}/"
        params = {"api_key": self.api_key}

        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk,
            "onstart": onstart,
        }

        if env:
            payload["env"] = env

        response = requests.put(url, headers=self.headers, params=params, json=payload)
        response.raise_for_status()
        return response.json()

    def get_instances(self) -> list:
        """내 인스턴스 목록 조회"""
        url = f"{self.BASE_URL}/instances/"
        params = {"api_key": self.api_key, "owner": "me"}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json().get("instances", [])


# ============================================================
# 인스턴스 선택 로직
# ============================================================

def filter_offers(offers: list, cfg: Config) -> list:
    """조건에 맞는 인스턴스 필터링"""
    filtered = []

    for offer in offers:
        price = offer.get("dph_total", 999)
        reliability = offer.get("reliability2", 0)
        disk_space = offer.get("disk_space", 0)
        gpu_name = offer.get("gpu_name", "")
        inet_down = offer.get("inet_down", 0)
        inet_up = offer.get("inet_up", 0)
        num_gpus = offer.get("num_gpus", 1)

        if (price <= cfg.max_price_per_hour and
            num_gpus == cfg.num_gpus and
            disk_space >= cfg.min_disk_space and
            gpu_name in cfg.preferred_gpus and
            inet_down >= cfg.min_inet_down and
            inet_up >= cfg.min_inet_up and
            offer.get("rentable", False)):
            filtered.append(offer)

    return filtered


def score_offer(offer: dict, cfg: Config) -> float:
    """인스턴스 점수 계산 (높을수록 좋음)"""
    score = 0.0

    price = offer.get("dph_total", 999)
    reliability = offer.get("reliability2", 0)

    # 가격 점수 (최대 30점)
    price_score = max(0, 30 * (1 - price / cfg.max_price_per_hour))
    score += price_score

    # 신뢰도 점수 (최대 15점)
    score += reliability * 15

    return score


def select_best_offer(offers: list, cfg: Config) -> Optional[dict]:
    """최적의 인스턴스 선택"""
    valid_offers = filter_offers(offers, cfg)

    if not valid_offers:
        return None

    scored = [(offer, score_offer(offer, cfg)) for offer in valid_offers]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[0][0]


# ============================================================
# 메인 실행
# ============================================================

def print_offer_info(offer: dict):
    """인스턴스 정보 출력"""
    print("\n" + "=" * 50)
    print("Selected Instance")
    print("=" * 50)
    print(f"  ID: {offer.get('id')}")
    print(f"  GPU: {offer.get('gpu_name')} x {offer.get('num_gpus', 1)}")
    print(f"  VRAM: {offer.get('gpu_ram', 0) / 1024:.1f} GB")
    print(f"  Price: ${offer.get('dph_total', 0):.3f}/hour")
    print(f"  Reliability: {offer.get('reliability2', 0) * 100:.1f}%")
    print(f"  Download: {offer.get('inet_down', 0):.0f} Mbps")
    print(f"  Upload: {offer.get('inet_up', 0):.0f} Mbps")
    print(f"  Location: {offer.get('geolocation', 'Unknown')}")
    print("=" * 50)


def build_env_vars(cfg: Config, instance_id: str = "") -> dict:
    """Docker 컨테이너에 전달할 환경변수 구성"""
    env = {
        "WANDB_API_KEY": cfg.env_vars.get("wandb_key", ""),
        "AWS_ACCESS_KEY_ID": cfg.env_vars.get("aws_key", ""),
        "AWS_SECRET_ACCESS_KEY": cfg.env_vars.get("aws_secret", ""),
        "AWS_DEFAULT_REGION": cfg.env_vars.get("aws_region", "ap-northeast-2"),
        "S3_DATA_PATH": cfg.env_vars.get("s3_path", ""),
        "VAST_API_KEY": cfg.api_key,
        "CONTAINER_ID": instance_id,
        "CONFIG_FILE": cfg.config_file,
    }
    return {k: v for k, v in env.items() if v}


def launch(cfg: Config, dry_run: bool = False) -> Optional[dict]:
    """메인 실행 함수"""
    print("Searching for instances on Vast.ai...")

    client = VastClient(cfg.api_key)

    # 1. 사용 가능한 인스턴스 검색
    try:
        offers = client.search_offers()
        print(f"Found {len(offers)} total offers")
    except Exception as e:
        print(f"Search failed: {e}")
        return None

    # 2. 최적 인스턴스 선택
    best = select_best_offer(offers, cfg)

    if not best:
        print("No instances matching the criteria found.")
        print("  -> Try relaxing requirements (price, GPU type, etc.)")
        return None

    print_offer_info(best)

    if dry_run:
        print("\n[DRY RUN] Skipping instance creation")
        return best

    # 3. 환경변수 구성
    env_vars = build_env_vars(cfg, str(best["id"]))

    # 4. 인스턴스 생성
    print(f"\nCreating instance...")
    print(f"  Image: {cfg.docker_image}")
    print(f"  Config: {cfg.config_file}")

    try:
        result = client.create_instance(
            offer_id=best["id"],
            image=cfg.docker_image,
            disk=cfg.disk_space,
            env=env_vars
        )
        instance_id = result.get('new_contract')
        print(f"\nInstance created successfully!")
        print(f"  Instance ID: {instance_id}")
        print(f"\nMonitor at: https://cloud.vast.ai/instances/")
        return result
    except Exception as e:
        print(f"Creation failed: {e}")
        return None


def list_my_instances(cfg: Config):
    """내 인스턴스 목록 조회"""
    client = VastClient(cfg.api_key)
    instances = client.get_instances()

    print(f"\nMy Instances ({len(instances)})")
    print("-" * 60)

    for inst in instances:
        status = inst.get("actual_status", "unknown")
        gpu = inst.get("gpu_name", "?")
        price = inst.get("dph_total", 0)
        print(f"  [{inst.get('id')}] {gpu} | ${price:.3f}/hr | {status}")


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Launch training on Vast.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Docker 이미지
    parser.add_argument(
        "--docker-image", "-i",
        default="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        help="Docker image to use"
    )

    # 학습 설정
    parser.add_argument(
        "--config", "-c",
        default="configs/config.yaml",
        help="Training config file path"
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra arguments for training script"
    )

    # 환경변수
    parser.add_argument("--wandb-key", default=os.getenv("WANDB_API_KEY", ""))
    parser.add_argument("--aws-key", default=os.getenv("AWS_ACCESS_KEY_ID", ""))
    parser.add_argument("--aws-secret", default=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    parser.add_argument("--aws-region", default=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))
    parser.add_argument("--s3-path", default=os.getenv("S3_DATA_PATH", ""))

    # Vast.ai 설정
    parser.add_argument(
        "--api-key",
        default=os.getenv("VAST_API_KEY", ""),
        help="Vast.ai API key"
    )
    parser.add_argument("--max-price", type=float, default=2.0)
    parser.add_argument("--min-reliability", type=float, default=0.9)
    parser.add_argument("--disk", type=float, default=50.0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-type",
        nargs="+",
        default=["RTX 4090"],
        help="Preferred GPU types"
    )

    # 동작 모드
    parser.add_argument("--dry-run", action="store_true", help="Search only, don't create")
    parser.add_argument("--list", action="store_true", help="List my instances")

    return parser.parse_args()


def main():
    args = parse_args()

    missing = []
    if not args.api_key:
        missing.append("VAST_API_KEY")
    if not args.wandb_key:
        missing.append("WANDB_API_KEY")

    if missing:
        print("Error: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print()
        print("Set them in your shell or create a .env file:")
        print("  export VAST_API_KEY=your_vastai_api_key")
        print("  export WANDB_API_KEY=your_wandb_api_key")
        sys.exit(1)

    cfg = Config(
        api_key=args.api_key,
        docker_image=args.docker_image,
        max_price_per_hour=args.max_price,
        min_reliability=args.min_reliability,
        disk_space=args.disk,
        num_gpus=args.gpus,
        preferred_gpus=tuple(args.gpu_type),
        config_file=args.config,
        extra_args=args.extra_args,
        env_vars={
            "wandb_key": args.wandb_key,
            "aws_key": args.aws_key,
            "aws_secret": args.aws_secret,
            "aws_region": args.aws_region,
            "s3_path": args.s3_path,
        }
    )

    if args.list:
        list_my_instances(cfg)
    else:
        result = launch(cfg, dry_run=args.dry_run)
        if result is None and not args.dry_run:
            sys.exit(1)


if __name__ == "__main__":
    main()
