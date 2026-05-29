"""TIGER-lite: encoder-decoder Transformer for generative recommendation.

입력: user behavior 시퀀스 (SID 토큰 + behavior 토큰 + position)
출력: 다음 아이템의 SID 시퀀스 (L 레벨, 자가회귀 생성)

SID 토큰화:
  각 SID 레벨 별로 vocab 이 다르므로 (e.g. 2048/1024/512), level offset 을 더해서
  단일 vocab 으로 합침. → tok_id_l = code_l + sum_{l'<l}(codebook_size_l')
  + 추가 special token: [PAD]=0, [BOS]=1, [SEP]=2, [EOS]=3 (앞에 reserve)

decoder input: <BOS> sid_0 sid_1 sid_2 ... <EOS>
decoder target: sid_0 sid_1 sid_2 ... <EOS>

PoC 1차: ~50M params (encoder 4 layer / decoder 4 layer / dim 384) — A6000 친화적.
v2: 200M 까지 확장 가능.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# special tokens (vocab head)
NUM_SPECIAL_TOKENS = 4
PAD_ID, BOS_ID, SEP_ID, EOS_ID = 0, 1, 2, 3


def build_sid_vocab(codebook_sizes: Tuple[int, ...]):
    """SID level offset 계산 (단일 vocab 으로 합쳐서 사용).

    Returns:
        offsets: level i 의 시작 토큰 id
        total_vocab: PAD/BOS/SEP/EOS + 모든 level codebook size 합
    """
    offsets = []
    cur = NUM_SPECIAL_TOKENS
    for size in codebook_sizes:
        offsets.append(cur)
        cur += size
    return offsets, cur


class TigerLiteNet(nn.Module):
    """단순화된 TIGER (encoder-decoder Transformer).

    Args:
        codebook_sizes:        SID 양자화 레벨별 크기 (sid_v2 와 일치)
        num_behaviors:         behavior 종류 수 (인코더 입력 토큰)
        max_seq_len:           encoder 최대 길이 (시퀀스 토큰 수 = items × levels)
        max_decode_len:        decoder 최대 길이 (= len(codebook_sizes) + 2 for BOS/EOS)
        dim:                   hidden 차원
        n_enc_layers / n_dec_layers / n_heads / ff_dim / dropout: Transformer 표준
    """

    def __init__(
        self,
        codebook_sizes: Tuple[int, ...] = (2048, 1024, 512),
        num_behaviors: int = 5,
        max_seq_len: int = 800,
        dim: int = 384,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 6,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        # === v7 신규 (백워드 호환: 0 → 미사용) ===
        max_order_positions: int = 0,   # 같은 ts(=같은 cart) 그룹 position 임베딩
    ):
        super().__init__()
        self.codebook_sizes = tuple(codebook_sizes)
        self.num_levels = len(codebook_sizes)
        self.offsets, self.total_vocab = build_sid_vocab(self.codebook_sizes)
        self.max_order_positions = max_order_positions

        # shared token embedding (encoder/decoder)
        self.tok_emb = nn.Embedding(self.total_vocab, dim, padding_idx=PAD_ID)
        self.behavior_emb = nn.Embedding(num_behaviors, dim, padding_idx=0)
        self.enc_pos_emb = nn.Embedding(max_seq_len, dim)
        self.dec_pos_emb = nn.Embedding(self.num_levels + 2, dim)  # BOS + L + EOS
        if max_order_positions > 0:
            self.order_pos_emb = nn.Embedding(max_order_positions, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # output head (tied with tok_emb 도 가능; 단순 분리)
        self.lm_head = nn.Linear(dim, self.total_vocab, bias=False)
        # mask: 각 decoder 위치(level)에서 valid token id 만 (다른 level token 무한 페널티)
        self.register_buffer("_level_mask", self._build_level_mask(), persistent=False)

    def _build_level_mask(self) -> torch.Tensor:
        """(num_levels + 1, total_vocab) — 위치별 유효 토큰 mask (1 = valid).

        position 0~num_levels-1: 해당 level codebook 만
        position num_levels: EOS 만
        """
        mask = torch.zeros(self.num_levels + 1, self.total_vocab, dtype=torch.bool)
        for l, (off, size) in enumerate(zip(self.offsets, self.codebook_sizes)):
            mask[l, off:off + size] = True
        mask[self.num_levels, EOS_ID] = True
        return mask

    def encode_sid(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, L) per-level int → (B, L) absolute token ids (offset 적용)."""
        offsets = torch.tensor(self.offsets, device=codes.device).unsqueeze(0)  # (1, L)
        return codes + offsets

    def forward(
        self,
        enc_item_tokens: torch.Tensor,    # (B, T_enc) — flattened SID tokens of history items
        enc_behavior_ids: torch.Tensor,   # (B, T_enc) — behavior per token (broadcast per item)
        enc_positions: torch.Tensor,      # (B, T_enc)
        enc_mask: torch.Tensor,           # (B, T_enc) bool (1 valid)
        dec_input_tokens: torch.Tensor,   # (B, T_dec)  e.g. [BOS, sid_0, sid_1, sid_2]
        dec_positions: torch.Tensor,      # (B, T_dec)
        enc_order_pos: Optional[torch.Tensor] = None,  # v7: (B, T_enc) — 같은 ts → 같은 position
    ) -> torch.Tensor:
        """returns logits (B, T_dec, total_vocab)."""
        # encoder
        enc_h = (
            self.tok_emb(enc_item_tokens)
            + self.behavior_emb(enc_behavior_ids)
            + self.enc_pos_emb(enc_positions.clamp(max=self.enc_pos_emb.num_embeddings - 1))
        )
        if self.max_order_positions > 0 and enc_order_pos is not None:
            enc_h = enc_h + self.order_pos_emb(
                enc_order_pos.clamp(max=self.max_order_positions - 1)
            )
        src_key_padding_mask = ~enc_mask.bool()
        memory = self.encoder(enc_h, src_key_padding_mask=src_key_padding_mask)

        # decoder
        dec_h = self.tok_emb(dec_input_tokens) + self.dec_pos_emb(dec_positions)
        T_dec = dec_input_tokens.size(1)
        causal = torch.triu(torch.ones(T_dec, T_dec, dtype=torch.bool, device=dec_h.device), diagonal=1)
        out = self.decoder(
            dec_h, memory,
            tgt_mask=causal,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.lm_head(out)  # (B, T_dec, V)
        return logits

    def apply_level_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """decoder 출력에 level-별 유효 토큰만 남기는 mask 적용.

        logits: (B, T_dec, V)
        """
        T = logits.size(1)
        # mask: (T, V)
        m = self._level_mask[:T]
        # invalid positions → -inf
        masked = logits.masked_fill(~m.unsqueeze(0), float("-inf"))
        return masked
