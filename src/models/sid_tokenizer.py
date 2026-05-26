"""SID Tokenizer LightningModule.

PLUM SIDv2 lineage:
  L = L_recon + α·L_vq + β·L_co_occurrence

학습 후 codebook 과 item→SID 매핑을 저장 (downstream HSTU/TIGER-lite 가 사용).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Metric

from nets.sid_v2 import RQVAE, ContentEncoder, co_occurrence_loss


class _LossAcc(Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, v: torch.Tensor):
        self.sum += v.detach()
        self.n += 1

    def compute(self):
        return self.sum / self.n.clamp(min=1)


class SIDTokenizerModel(L.LightningModule):
    """Content encoder → RQ-VAE tokenizer.

    Args:
        latent_dim:     RQ-VAE latent 차원
        codebook_sizes: multi-resolution codebook (PLUM 기본 2048/2/...)
        text_dim, image_dim: 사전학습 임베딩 차원 (0 이면 미사용)
        cat_emb_dim, brand_emb_dim, price_emb_dim: lookup embedding 차원
        num_price_buckets: 가격 분위 수 (0 이면 미사용)
        recon_weight, vq_weight, co_weight: 손실 가중
        co_temperature: contrastive temperature
        encoder_hidden, decoder_hidden, dropout: MLP 구조
        commitment_cost, ema_decay: VQ-VAE 표준 파라미터
        lr, weight_decay: optimizer
    """

    def __init__(
        self,
        latent_dim: int = 256,
        codebook_sizes: tuple = (2048, 1024, 512),
        text_dim: int = 512,
        image_dim: int = 0,
        cat_emb_dim: int = 64,
        brand_emb_dim: int = 32,
        num_price_buckets: int = 10,
        price_emb_dim: int = 16,
        content_out_dim: int = 512,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        co_weight: float = 0.5,
        co_temperature: float = 0.07,
        encoder_hidden: tuple = (512, 256),
        decoder_hidden: tuple = (256, 512),
        dropout: float = 0.0,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # net은 setup()에서 DataModule로부터 vocab 크기 받아 초기화
        self.encoder: Optional[ContentEncoder] = None
        self.vae: Optional[RQVAE] = None
        # metrics
        self.train_recon = _LossAcc()
        self.train_vq = _LossAcc()
        self.train_co = _LossAcc()
        self.val_recon = _LossAcc()

    def setup(self, stage: str):
        if self.encoder is not None:
            return
        dm = self.trainer.datamodule
        self.encoder = ContentEncoder(
            text_dim=self.hparams.text_dim,
            image_dim=self.hparams.image_dim,
            num_categories=dm.num_categories,
            num_brands=dm.num_brands,
            cat_emb_dim=self.hparams.cat_emb_dim,
            brand_emb_dim=self.hparams.brand_emb_dim,
            num_price_buckets=self.hparams.num_price_buckets,
            price_emb_dim=self.hparams.price_emb_dim,
            out_dim=self.hparams.content_out_dim,
        )
        self.vae = RQVAE(
            input_dim=self.hparams.content_out_dim,
            latent_dim=self.hparams.latent_dim,
            codebook_sizes=tuple(self.hparams.codebook_sizes),
            encoder_hidden=tuple(self.hparams.encoder_hidden),
            decoder_hidden=tuple(self.hparams.decoder_hidden),
            dropout=self.hparams.dropout,
            commitment_cost=self.hparams.commitment_cost,
            ema_decay=self.hparams.ema_decay,
        )

    def _encode_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            text_emb=batch.get("text_emb"),
            image_emb=batch.get("image_emb"),
            category_id=batch.get("category_id"),
            brand_id=batch.get("brand_id"),
            price_bucket=batch.get("price_bucket"),
        )

    def training_step(self, batch, batch_idx):
        # batch: {anchor: {...features...}, positive: {...features...}}
        anchor_x = self._encode_batch(batch["anchor"])
        positive_x = self._encode_batch(batch["positive"])

        # RQ-VAE forward on both (codebook is shared)
        out_a = self.vae(anchor_x)
        out_p = self.vae(positive_x)

        # 1) reconstruction
        recon = (
            F.mse_loss(out_a["x_recon"], anchor_x.detach())
            + F.mse_loss(out_p["x_recon"], positive_x.detach())
        ) * 0.5

        # 2) vq
        vq = (out_a["vq_loss"] + out_p["vq_loss"]) * 0.5

        # 3) co-occurrence on quantized
        co = co_occurrence_loss(out_a["q"], out_p["q"], temperature=self.hparams.co_temperature)

        loss = (
            self.hparams.recon_weight * recon
            + self.hparams.vq_weight * vq
            + self.hparams.co_weight * co
        )
        self.train_recon(recon)
        self.train_vq(vq)
        self.train_co(co)
        return loss

    def on_train_epoch_end(self):
        self.log("train/recon", self.train_recon.compute(), prog_bar=True)
        self.log("train/vq", self.train_vq.compute(), prog_bar=False)
        self.log("train/co", self.train_co.compute(), prog_bar=True)
        self.train_recon.reset()
        self.train_vq.reset()
        self.train_co.reset()

    def validation_step(self, batch, batch_idx):
        # validation: 전체 items 의 SID 분포 / recon 만 측정
        x = self._encode_batch(batch)
        out = self.vae(x)
        self.val_recon(F.mse_loss(out["x_recon"], x.detach()))
        # cache codes 로 SID uniqueness 누적 계산
        if not hasattr(self, "_val_codes_buf"):
            self._val_codes_buf = []
        self._val_codes_buf.append(out["codes"].detach().cpu())

    def on_validation_epoch_end(self):
        self.log("val/recon", self.val_recon.compute(), prog_bar=True)
        self.val_recon.reset()
        if hasattr(self, "_val_codes_buf") and self._val_codes_buf:
            all_codes = torch.cat(self._val_codes_buf, dim=0)
            self._val_codes_buf = []
            n_total = all_codes.size(0)
            n_unique = len({tuple(c.tolist()) for c in all_codes})
            self.log("val/sid_uniqueness", n_unique / max(n_total, 1), prog_bar=True)
        util = self.vae.quantizer.code_utilization()
        for i, u in enumerate(util):
            self.log(f"val/code_util_lvl{i}", u, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
