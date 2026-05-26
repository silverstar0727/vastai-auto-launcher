"""SID Tokenizer 학습 완료 후 active items 전체의 SID 매핑 + codebook 저장.

산출 (PoC v2_full data root 아래):
  meta/item_to_sid.parquet   - goods_sno, sid_0, sid_1, ..., sid_{L-1}  (TIGER-lite 가 사용)
  meta/sid_codebooks.pt      - {f'codebook_{i}': tensor (n, D)} - 디버그용
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L


class SaveSIDMapCallback(L.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # rank0 only
        if trainer.global_rank != 0:
            return

        dm = trainer.datamodule
        store = dm.store
        active_snos = dm.active_snos
        device = pl_module.device

        encoder = pl_module.encoder
        vae = pl_module.vae
        encoder.eval(); vae.eval()

        # 전체 active item 의 SID 추출 (batch)
        codes_all = []
        snos_all = []
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, len(active_snos), batch_size):
                snos = active_snos[i:i + batch_size]
                # 각 item 의 feature 모으기
                feats = [store.get(int(s)) for s in snos]
                batch = {
                    "text_emb": torch.stack([f["text_emb"] for f in feats]).to(device),
                    "category_id": torch.stack([f["category_id"] for f in feats]).to(device),
                    "brand_id": torch.stack([f["brand_id"] for f in feats]).to(device),
                    "price_bucket": torch.stack([f["price_bucket"] for f in feats]).to(device),
                }
                x = encoder(
                    text_emb=batch["text_emb"],
                    category_id=batch["category_id"],
                    brand_id=batch["brand_id"],
                    price_bucket=batch["price_bucket"],
                )
                codes = vae.encode_to_sid(x).cpu().numpy()
                codes_all.append(codes)
                snos_all.extend(snos.tolist())

        all_codes = np.concatenate(codes_all, axis=0)
        L_ = all_codes.shape[1]
        df = pd.DataFrame({"goods_sno": snos_all})
        for l in range(L_):
            df[f"sid_{l}"] = all_codes[:, l]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = self.output_dir / "item_to_sid.parquet"
        df.to_parquet(out_parquet, index=False)
        print(f"[SaveSIDMapCallback] {out_parquet} ({len(df):,} rows)")

        # codebook 도 저장 (디버그용)
        codebooks = {}
        for i in range(vae.quantizer.num_levels):
            codebooks[f"codebook_{i}"] = getattr(vae.quantizer, f"codebook_{i}").cpu()
        out_pt = self.output_dir / "sid_codebooks.pt"
        torch.save(codebooks, out_pt)
        print(f"[SaveSIDMapCallback] {out_pt}")

        # SID uniqueness
        n_unique = len(set(map(tuple, all_codes.tolist())))
        print(f"[SaveSIDMapCallback] SID uniqueness: {n_unique:,} / {len(df):,} "
              f"({n_unique/len(df)*100:.2f}%)")
