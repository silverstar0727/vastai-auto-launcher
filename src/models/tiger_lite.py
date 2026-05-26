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
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net: TigerLiteNet = None
        self._beam: ConstrainedBeamSearch = None
        self._trie: SIDTrie = None
        self.loss_acc = _Acc()

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
        )
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
        self.loss_acc(loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train/loss", self.loss_acc.compute(), prog_bar=True)
        self.loss_acc.reset()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # encoder pass
        enc_h = (
            self.net.tok_emb(batch["enc_tokens"])
            + self.net.behavior_emb(batch["enc_beh"])
            + self.net.enc_pos_emb(batch["enc_pos"].clamp(max=self.net.enc_pos_emb.num_embeddings - 1))
        )
        src_key_padding_mask = ~batch["enc_mask"].bool()
        memory = self.net.encoder(enc_h, src_key_padding_mask=src_key_padding_mask)

        beams = self._beam.search(memory, src_key_padding_mask)
        # beam → target match Recall@K
        target = batch["dec_target"][:, :-1]  # exclude EOS, (B, L)
        target_list = [tuple(t.tolist()) for t in target]

        for k in self.hparams.eval_ks:
            hits = 0
            total = 0
            for user_beams, tgt in zip(beams, target_list):
                topk = [tuple(p) for p, _ in user_beams[:k]]
                if tgt in topk:
                    hits += 1
                total += 1
            self.log(f"val/Recall@{k}", hits / max(total, 1),
                     prog_bar=(k == 10), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
