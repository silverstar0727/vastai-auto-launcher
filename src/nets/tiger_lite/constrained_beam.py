"""Inventory-aware constrained beam search.

active item 의 SID set 을 trie 로 구축 → 디코딩 시 valid prefix 만 허용 → hallucination 0% 보장.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class SIDTrie:
    """active item 의 SID prefix trie. 각 노드는 다음 가능 토큰 set 을 담음.

    Args:
        item_sids: (N, L) int — 각 행이 한 아이템의 SID tuple (absolute token ids 포함)
    """

    def __init__(self, item_sids: torch.Tensor):
        # store as nested dict for simplicity (small overhead since L=3 or 4)
        self.root: Dict[int, dict] = {}
        for row in item_sids.tolist():
            node = self.root
            for tok in row:
                node = node.setdefault(tok, {})
            node["__END__"] = True

    def valid_next(self, prefix: List[int]) -> List[int]:
        node = self.root
        for tok in prefix:
            if tok not in node:
                return []
            node = node[tok]
        return [k for k in node.keys() if k != "__END__"]


class ConstrainedBeamSearch:
    """디코딩 시 trie 와 level-mask 모두 적용.

    Args:
        model: TigerLiteNet
        trie:  SIDTrie
        beam_width:    K
        max_len:       L (decoder 길이 - BOS 제외)
        end_token:     EOS_ID
    """

    def __init__(
        self,
        model,
        trie: SIDTrie,
        beam_width: int = 32,
        max_len: int = 3,
        bos_token: int = 1,
        eos_token: int = 3,
    ):
        self.model = model
        self.trie = trie
        self.beam_width = beam_width
        self.max_len = max_len
        self.bos = bos_token
        self.eos = eos_token

    @torch.no_grad()
    def search(self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> List[List[Tuple[List[int], float]]]:
        """
        Args:
            memory: (B, T_enc, D) encoder 출력
            memory_key_padding_mask: (B, T_enc)
        Returns:
            beams: B 개 user 의 beam 결과. 각 beam = (sid_seq[1..L], log_prob).
        """
        B = memory.size(0)
        device = memory.device
        # beam 초기화: (B, beam, L+1) — 첫 위치 BOS
        beams = [[([self.bos], 0.0)] for _ in range(B)]

        for step in range(self.max_len):
            new_beams: List[List[Tuple[List[int], float]]] = [[] for _ in range(B)]
            for b in range(B):
                for prefix, logp in beams[b]:
                    valid = self.trie.valid_next(prefix[1:])  # BOS 제외한 prefix
                    if not valid:
                        # 끝났거나 dead end — 그대로 유지 (점수 변동 없음)
                        new_beams[b].append((prefix, logp))
                        continue
                    # decode 1 step
                    dec_input = torch.tensor([prefix], device=device)
                    dec_positions = torch.arange(len(prefix), device=device).unsqueeze(0)
                    enc_h_b = memory[b:b+1]
                    mem_mask = memory_key_padding_mask[b:b+1]
                    dec_h = self.model.tok_emb(dec_input) + self.model.dec_pos_emb(dec_positions)
                    T = dec_input.size(1)
                    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
                    out = self.model.decoder(dec_h, enc_h_b, tgt_mask=causal, memory_key_padding_mask=mem_mask)
                    logits = self.model.lm_head(out[:, -1])  # (1, V)
                    # mask invalid (level + trie)
                    invalid = torch.ones_like(logits, dtype=torch.bool)
                    valid_t = torch.tensor(valid, device=device)
                    invalid.scatter_(1, valid_t.unsqueeze(0), False)
                    logits = logits.masked_fill(invalid, float("-inf"))
                    logp_t = torch.log_softmax(logits, dim=-1).squeeze(0)
                    topk = min(self.beam_width, len(valid))
                    top_vals, top_idx = logp_t.topk(topk)
                    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
                        new_beams[b].append((prefix + [i], logp + v))
                # prune beam
                new_beams[b].sort(key=lambda x: -x[1])
                new_beams[b] = new_beams[b][:self.beam_width]
            beams = new_beams

        # 정렬된 beam (BOS 제외) 반환
        return [[(p[1:], lp) for p, lp in user_beams] for user_beams in beams]
