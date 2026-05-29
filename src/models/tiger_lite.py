"""TIGER-lite generative recommendation LightningModule.

학습: encoder(user history SID 시퀀스) → decoder(next item SID 자가회귀 생성).
평가: constrained beam search → Recall@K, NDCG@K (디코딩된 SID → item 복원).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Metric

from nets.tiger_lite import TigerLiteNet, SIDTrie, ConstrainedBeamSearch
from optimizers.linear_warmup import LinearWarmupCosineAnnealingLR


BEHAVIOR_WEIGHTS: Dict[int, float] = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 20.0}


class _Acc(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("s", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, v):
        self.s += v.detach(); self.n += 1
    def compute(self):
        return self.s / self.n.clamp(min=1)


class TIGERLiteModel(L.LightningModule):
    def __init__(
        self,
        dim: int = 384,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 6,
        ff_dim: int = 1536,
        max_seq_len: int = 800,
        dropout: float = 0.1,
        # eval
        beam_width: int = 32,
        eval_ks: tuple = (10, 50),
        # optim
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_epochs: int = 3,
        eta_min: float = 1e-5,
        # === v7 신규 ===
        max_order_positions: int = 0,    # 0 → order pos 미사용 (기존 동작)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net: TigerLiteNet = None
        self._beam: ConstrainedBeamSearch = None
        self._trie: SIDTrie = None
        self.loss_acc = _Acc()
        self._val_topk_items = set()  # Coverage@K 측정용 (epoch 별 reset)

    def setup(self, stage: str):
        if self.net is not None:
            return
        dm = self.trainer.datamodule
        self.net = TigerLiteNet(
            codebook_sizes=tuple(dm.hparams.codebook_sizes),
            num_behaviors=dm.num_behaviors,
            max_seq_len=dm.hparams.max_history_items * dm.store.L + 4,  # +pad
            dim=self.hparams.dim,
            n_enc_layers=self.hparams.n_enc_layers,
            n_dec_layers=self.hparams.n_dec_layers,
            n_heads=self.hparams.n_heads,
            ff_dim=self.hparams.ff_dim,
            dropout=self.hparams.dropout,
            max_order_positions=self.hparams.max_order_positions,
        )
        self._num_total_items = len(dm.store.sno_to_sid) if hasattr(dm.store, "sno_to_sid") else 0
        # trie for inventory-aware decoding
        all_sids = dm.store.all_item_sids()
        self._trie = SIDTrie(all_sids)
        self._beam = ConstrainedBeamSearch(
            model=self.net, trie=self._trie,
            beam_width=self.hparams.beam_width,
            max_len=dm.store.L,
        )

    def training_step(self, batch, batch_idx):
        logits = self.net(
            batch["enc_tokens"], batch["enc_beh"], batch["enc_pos"], batch["enc_mask"],
            batch["dec_input"], batch["dec_pos"],
            enc_order_pos=batch.get("enc_order_pos"),
        )  # (B, T_dec, V)
        target = batch["dec_target"]  # (B, T_dec)
        # behavior weighted CE (sample 별 가중)
        loss_per = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=0, reduction="none",
        ).view(target.size())  # (B, T_dec)
        # 평균 per sample
        per_sample = loss_per.mean(dim=1)
        weights = torch.tensor(
            [BEHAVIOR_WEIGHTS.get(b.item(), 1.0) for b in batch["target_behavior"]],
            device=logits.device,
        )
        loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=target.size(0))
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                 on_step=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        import math
        # encoder pass
        enc_h = (
            self.net.tok_emb(batch["enc_tokens"])
            + self.net.behavior_emb(batch["enc_beh"])
            + self.net.enc_pos_emb(batch["enc_pos"].clamp(max=self.net.enc_pos_emb.num_embeddings - 1))
        )
        if self.hparams.max_order_positions > 0 and "enc_order_pos" in batch:
            enc_h = enc_h + self.net.order_pos_emb(
                batch["enc_order_pos"].clamp(max=self.net.order_pos_emb.num_embeddings - 1)
            )
        src_key_padding_mask = ~batch["enc_mask"].bool()
        memory = self.net.encoder(enc_h, src_key_padding_mask=src_key_padding_mask)

        beams = self._beam.search(memory, src_key_padding_mask)
        target = batch["dec_target"][:, :-1]  # exclude EOS, (B, L)
        target_list = [tuple(t.tolist()) for t in target]

        # Recall@K, NDCG@K, Coverage@K (모든 k 한 번에)
        for k in self.hparams.eval_ks:
            hits = 0
            ndcg_sum = 0.0
            total = 0
            for user_beams, tgt in zip(beams, target_list):
                topk = [tuple(p) for p, _ in user_beams[:k]]
                self._val_topk_items.update(topk)  # Coverage 측정용
                if tgt in topk:
                    hits += 1
                    rank = topk.index(tgt)  # 0-based
                    ndcg_sum += 1.0 / math.log2(rank + 2)  # gain=1, log2(rank+2)
                total += 1
            self.log(f"val/Recall@{k}", hits / max(total, 1),
                     prog_bar=(k == 10), on_step=False, on_epoch=True)
            self.log(f"val/NDCG@{k}", ndcg_sum / max(total, 1),
                     prog_bar=(k == 10), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # Coverage@K = unique items 본 / 전체 item 수 (top-K 의 union)
        if self._num_total_items > 0 and len(self._val_topk_items) > 0:
            cov = len(self._val_topk_items) / self._num_total_items
            self.log("val/Coverage", cov, prog_bar=False)
        self._val_topk_items = set()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # step 단위 warmup+cosine. epoch 단위 스케줄러는 epoch 0 내내 lr=0 → 학습 안 됨.
        from transformers import get_cosine_schedule_with_warmup
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = min(1500, max(100, total_steps // 100))
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        print(f"[optim] total_steps={total_steps} warmup_steps={warmup_steps} "
              f"peak_lr={self.hparams.lr}", flush=True)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
