"""SID Tokenizer LightningModule with LETTER L_CF + L_Div regularizers.

LETTER (Wang et al., CIKM 2024, arXiv:2405.07314) 의 RQ-VAE 정규화를 추가:
  L = L_recon + α·L_vq + β·L_co(optional) + α_cf·L_CF + α_div·L_Div

기존 `SIDTokenizerModel` 을 상속하며, 다음을 가짐:
  - RQVAE → RQVAELetter 로 교체 (setup 단계).
  - DataModule 의 `batch["anchor"]["cf_emb"]` 를 받아 L_CF 계산.
  - co_occurrence_loss 는 `co_weight` 로 끄거나 유지 가능 (paper 그대로 가려면 0).

기존 `SIDTokenizerModel` 학습 경로 무변경 — 본 모듈을 사용하지 않으면 기존 동작 그대로.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nets.sid_v2 import ContentEncoder
from nets.sid_v2.rq_vae_letter import RQVAELetter
from .sid_tokenizer import SIDTokenizerModel


class SIDTokenizerLetterModel(SIDTokenizerModel):
    """SID Tokenizer + LETTER L_CF + L_Div."""

    def __init__(
        self,
        # LETTER 신규 인자
        cf_emb_dim: int = 64,
        cf_weight: float = 0.02,            # paper best α = 0.02
        div_weight: float = 1e-4,           # paper best β ≈ 1e-4
        div_clusters: int = 10,
        div_recluster_every: int = 500,
        **kwargs,
    ):
        # 부모 init (기존 sid_tokenizer 인자 모두 수용)
        super().__init__(**kwargs)
        # save_hyperparameters 가 부모에서 호출되어 신규 인자는 따로 저장
        self.save_hyperparameters(
            "cf_emb_dim", "cf_weight", "div_weight", "div_clusters", "div_recluster_every"
        )

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
        self.vae = RQVAELetter(
            input_dim=self.hparams.content_out_dim,
            latent_dim=self.hparams.latent_dim,
            codebook_sizes=tuple(self.hparams.codebook_sizes),
            encoder_hidden=tuple(self.hparams.encoder_hidden),
            decoder_hidden=tuple(self.hparams.decoder_hidden),
            dropout=self.hparams.dropout,
            commitment_cost=self.hparams.commitment_cost,
            ema_decay=self.hparams.ema_decay,
            # LETTER specific
            cf_emb_dim=self.hparams.cf_emb_dim,
            letter_div_clusters=self.hparams.div_clusters,
            letter_div_recluster_every=self.hparams.div_recluster_every,
        )

    def training_step(self, batch, batch_idx):
        from nets.sid_v2 import co_occurrence_loss

        anchor_x = self._encode_batch(batch["anchor"])
        positive_x = self._encode_batch(batch["positive"])
        out_a = self.vae(anchor_x)
        out_p = self.vae(positive_x)

        recon = (
            F.mse_loss(out_a["x_recon"], anchor_x.detach())
            + F.mse_loss(out_p["x_recon"], positive_x.detach())
        ) * 0.5
        vq = (out_a["vq_loss"] + out_p["vq_loss"]) * 0.5
        co = co_occurrence_loss(
            out_a["q"], out_p["q"], temperature=self.hparams.co_temperature
        )

        # LETTER L_CF — anchor 측 quantized vs CF teacher embedding
        cf_emb = batch["anchor"].get("cf_emb")
        if cf_emb is None:
            raise ValueError(
                "[SIDTokenizerLetterModel] batch['anchor']['cf_emb'] 가 필요합니다 — "
                "DataModule 이 CF teacher embedding 을 로드하는지 확인하세요."
            )
        loss_cf = self.vae.compute_cf_loss(out_a["q"], cf_emb)

        # LETTER L_Div — codebook 자체에 대한 정규화 (batch 무관)
        loss_div = self.vae.compute_div_loss()

        loss = (
            self.hparams.recon_weight * recon
            + self.hparams.vq_weight * vq
            + self.hparams.co_weight * co
            + self.hparams.cf_weight * loss_cf
            + self.hparams.div_weight * loss_div
        )
        self.train_recon(recon)
        self.train_vq(vq)
        self.train_co(co)
        self.log("train/cf", loss_cf, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/div", loss_div, on_step=True, on_epoch=True, prog_bar=False)
        return loss
