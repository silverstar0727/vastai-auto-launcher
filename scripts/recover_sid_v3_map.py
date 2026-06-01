"""SID v3 학습이 epoch 33 까지 진행됐고 best ckpt 가 저장된 상태에서,
on_fit_end callback (SaveSIDMapCallback) 가 attribute_ids 누락으로 실패한 케이스를
복구하기 위해 standalone 으로 SID map (meta_v3/item_to_sid.parquet) 을 생성한다.

사용:
    python scripts/recover_sid_v3_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets.sid_tokenizer import SIDTokenizerDataModule  # noqa: E402
from models.sid_tokenizer import SIDTokenizerModel  # noqa: E402


def main():
    cfg_path = ROOT / "configs" / "sid_v2_v3.yaml"
    ckpt_path = ROOT / "logs" / "sid_v2" / "v3" / "fit" / "checkpoints" / "best_sid_tokenizer_v3.ckpt"
    out_dir = Path("/home/jeongmindo/projects/ably/data/reco_experiments/v2_full/meta_v3")

    print(f"[recover] config: {cfg_path}")
    print(f"[recover] ckpt:   {ckpt_path}")
    print(f"[recover] out:    {out_dir}")
    assert ckpt_path.exists(), f"ckpt 없음: {ckpt_path}"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # DataModule
    dm_kwargs = cfg["data"]["init_args"]
    dm = SIDTokenizerDataModule(**dm_kwargs)
    dm.prepare_data()
    dm.setup("fit")
    print(f"[recover] datamodule ready, active items: {len(dm.active_snos):,}")
    print(f"[recover] num_attribute_values: {dm.num_attribute_values}")

    # Model — config 의 init_args 로 instantiate (encoder/vae 는 lazy → 수동 trigger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = dict(cfg["model"]["init_args"])
    model = SIDTokenizerModel(**model_kwargs)

    # setup() 흉내 — fake trainer 로 datamodule 주입 후 호출
    class _FakeTrainer:
        def __init__(self, dm):
            self.datamodule = dm
    # Lightning 의 self.trainer 는 _trainer property, _setup_trainer/_trainer attr 로 우회
    # 가장 간단: model.setup() 의 내부 로직을 직접 복제
    from nets.sid_v2 import ContentEncoder, RQVAE
    num_attr = dm.num_attribute_values
    model.encoder = ContentEncoder(
        text_dim=model.hparams.text_dim,
        image_dim=model.hparams.image_dim,
        num_categories=dm.num_categories,
        num_brands=dm.num_brands,
        cat_emb_dim=model.hparams.cat_emb_dim,
        brand_emb_dim=model.hparams.brand_emb_dim,
        num_price_buckets=model.hparams.num_price_buckets,
        price_emb_dim=model.hparams.price_emb_dim,
        out_dim=model.hparams.content_out_dim,
        num_attribute_values=num_attr,
        attribute_emb_dim=model.hparams.attribute_emb_dim,
        max_attributes_per_item=model.hparams.max_attributes_per_item,
    )
    model.vae = RQVAE(
        input_dim=model.hparams.content_out_dim,
        latent_dim=model.hparams.latent_dim,
        codebook_sizes=tuple(model.hparams.codebook_sizes),
        encoder_hidden=tuple(model.hparams.encoder_hidden),
        decoder_hidden=tuple(model.hparams.decoder_hidden),
        dropout=model.hparams.dropout,
        commitment_cost=model.hparams.commitment_cost,
        ema_decay=model.hparams.ema_decay,
    )
    if model.hparams.aux_category_weight > 0 and dm.num_categories > 0:
        import torch.nn as nn
        model.aux_cat_head = nn.Linear(model.hparams.content_out_dim, dm.num_categories)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[recover] state_dict — missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"  unexpected[:5]: {unexpected[:5]}")
    model.eval().to(device)

    encoder = model.encoder
    vae = model.vae
    store = dm.store
    active_snos = dm.active_snos

    # SID 추출
    codes_all = []
    snos_all = []
    batch_size = 8192
    with torch.no_grad():
        for i in range(0, len(active_snos), batch_size):
            snos = active_snos[i : i + batch_size]
            feats = [store.get(int(s)) for s in snos]
            text_emb = torch.stack([f["text_emb"] for f in feats]).to(device)
            category_id = torch.stack([f["category_id"] for f in feats]).to(device)
            brand_id = torch.stack([f["brand_id"] for f in feats]).to(device)
            price_bucket = torch.stack([f["price_bucket"] for f in feats]).to(device)
            attribute_ids = None
            if "attribute_ids" in feats[0]:
                attribute_ids = torch.stack([f["attribute_ids"] for f in feats]).to(device)
            x = encoder(
                text_emb=text_emb,
                category_id=category_id,
                brand_id=brand_id,
                price_bucket=price_bucket,
                attribute_ids=attribute_ids,
            )
            codes = vae.encode_to_sid(x).cpu().numpy()
            codes_all.append(codes)
            snos_all.extend(snos.tolist())
            if (i // batch_size) % 10 == 0:
                print(f"[recover] {i + len(snos):,} / {len(active_snos):,}", flush=True)

    all_codes = np.concatenate(codes_all, axis=0)
    L_ = all_codes.shape[1]
    df = pd.DataFrame({"goods_sno": snos_all})
    for l in range(L_):
        df[f"sid_{l}"] = all_codes[:, l]

    sid_cols = [f"sid_{l}" for l in range(L_)]
    df[f"sid_{L_}"] = df.groupby(sid_cols).cumcount().astype("int64")
    max_collision = int(df[f"sid_{L_}"].max())
    n_collided = int((df[f"sid_{L_}"] > 0).sum())
    print(
        f"[recover] collision token sid_{L_}: max={max_collision}, "
        f"collided={n_collided:,} ({n_collided / len(df) * 100:.2f}%)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "item_to_sid.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[recover] saved: {out_parquet} ({len(df):,} rows, SID 길이={L_ + 1})")

    codebooks = {}
    for i in range(vae.quantizer.num_levels):
        codebooks[f"codebook_{i}"] = getattr(vae.quantizer, f"codebook_{i}").cpu()
    out_pt = out_dir / "sid_codebooks.pt"
    torch.save(codebooks, out_pt)
    print(f"[recover] saved: {out_pt}")

    n_unique = len(set(map(tuple, all_codes.tolist())))
    print(
        f"[recover] SID uniqueness: {n_unique:,} / {len(df):,} "
        f"({n_unique / len(df) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
