"""TIGER-Gemma LightningModule.

두 stage 지원 (config 의 `stage` 로 선택):
  - 'cpt' : user seq + item meta next-token LM
  - 'sft' : (user history → next-SID tuple) generation,
            검증에 constrained beam search 적용

CPT 결과 ckpt 를 SFT 시작 시 ckpt_path 로 로드 (Lightning CLI 표준 방식).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Metric

from nets.tiger_lite.gemma_backbone import GemmaTigerBackbone
from optimizers.linear_warmup import LinearWarmupCosineAnnealingLR


class _Acc(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("s", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, v):
        self.s += v.detach()
        self.n += 1
    def compute(self):
        return self.s / self.n.clamp(min=1)


class TIGERGemmaModel(L.LightningModule):
    """
    Args:
        stage: 'cpt' or 'sft'
        model_name: HF model id
        codebook_sizes: SID 레벨별 코드북 크기 (sid_v2 와 동일)
        lora_r/alpha/dropout, target_modules: LoRA hparam
        load_in_4bit: 양자화 (메모리 절약)
        lr, weight_decay, warmup_epochs, eta_min
    """

    def __init__(
        self,
        stage: str = "cpt",
        model_name: str = "google/gemma-3-4b-it",
        codebook_sizes: tuple = (2048, 1024, 512),
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        target_modules_regex: str = (
            r".*language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(gate_proj|up_proj|down_proj))$"
        ),
        load_in_4bit: bool = False,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,     # 전체 step의 3%를 warmup (step 단위)
        eta_min: float = 1e-6,
        # SFT 전용
        beam_width: int = 32,
        eval_ks: tuple = (10, 50),
    ):
        super().__init__()
        self.save_hyperparameters()
        assert stage in ("cpt", "sft")
        self.backbone: Optional[GemmaTigerBackbone] = None
        self.loss_acc = _Acc()
        self.val_loss_acc = _Acc()

    def configure_model(self):
        if self.backbone is not None:
            return
        self.backbone = GemmaTigerBackbone(
            model_name=self.hparams.model_name,
            codebook_sizes=tuple(self.hparams.codebook_sizes),
            lora_r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules_regex=self.hparams.target_modules_regex,
            load_in_4bit=self.hparams.load_in_4bit,
        )
        info = self.backbone.trainable_params_count()
        print(f"[TIGERGemma] params: total={info['total']:,} "
              f"trainable={info['trainable']:,} ({info['trainable_pct']:.2f}%)")

    def setup(self, stage: str):
        self.configure_model()
        # DataModule 에 tokenizer 전달
        if hasattr(self.trainer.datamodule, "set_tokenizer"):
            self.trainer.datamodule.set_tokenizer(self.backbone.tokenizer)

    # ---------- CPT ----------

    def _step_cpt(self, batch):
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return out.loss

    # ---------- SFT (next-SID generation) ----------

    def _step_sft(self, batch):
        # SFT 데이터셋은 input_ids 가 "user_history <BOS_SID> sid_0 sid_1 sid_2 <EOS_SID>" 형태
        # labels 는 마지막 4 토큰만 (BOS 이후) 학습. 다른 위치 -100.
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return out.loss

    def training_step(self, batch, batch_idx):
        if self.hparams.stage == "cpt":
            loss = self._step_cpt(batch)
        else:
            loss = self._step_sft(batch)
        self.loss_acc(loss)
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/loss", self.loss_acc.compute(), prog_bar=True)
        self.loss_acc.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.hparams.stage == "cpt":
            loss = self._step_cpt(batch)
        else:
            loss = self._step_sft(batch)
        self.val_loss_acc(loss)

    def on_validation_epoch_end(self):
        v = self.val_loss_acc.compute()
        self.log("val/loss", v, prog_bar=True)
        # perplexity (CPT 에 유의미)
        self.log("val/ppl", torch.exp(v).clamp(max=1e6), prog_bar=True)
        self.val_loss_acc.reset()

    def configure_optimizers(self):
        # LoRA + new token embeddings 만 trainable
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        # step 단위 warmup + cosine decay (epoch-based 면 1 epoch=수일이라 lr=0 문제)
        from transformers import get_cosine_schedule_with_warmup
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(10, int(self.hparams.warmup_ratio * total_steps))
        print(f"[TIGERGemma] scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}")
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
