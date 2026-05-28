"""Gemma 4 E4B backbone for TIGER-lite (PLUM-inspired generative recsys).

PLUM (arXiv 2510.07784) 패턴:
  - 사전학습 LLM (Gemma) 의 token embedding 공간에 우리 SID 토큰을 추가
  - CPT (Continued Pre-Training) 로 user behavior + item metadata next-token prediction
  - SFT (Supervised Fine-Tuning) 로 next-SID generation

본 구현은:
  - Gemma 4 E4B (multimodal) 의 text-only mode 사용 (Gemma4ForCausalLM)
  - LoRA (rank=32, alpha=64) on attention + MLP
  - native bfloat16 (Gemma 4 표준 dtype)
  - text_config: hidden=2560, layers=42, heads=8, vocab=262K
  - 새 special tokens: SID_0_{0..2047}, SID_1_{0..1023}, SID_2_{0..511}
                       + <BEH_click>, <BEH_like>, <BEH_cart>, <BEH_purchase>
                       + <SEP>, <BOS_SID>, <EOS_SID>
  - 새 토큰 embedding 은 학습 가능 (LoRA + new embed both trainable)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer


# new special tokens for SID + behavior
def make_sid_tokens(codebook_sizes: Tuple[int, ...] = (2048, 1024, 512)) -> List[str]:
    """SID 레벨 × code 별 새 special token 이름 생성."""
    toks = []
    for level, n in enumerate(codebook_sizes):
        for i in range(n):
            toks.append(f"<SID_{level}_{i}>")
    return toks


BEHAVIOR_TOKENS = ["<BEH_click>", "<BEH_like>", "<BEH_cart>", "<BEH_purchase>"]
STRUCTURAL_TOKENS = ["<SEP_ITEM>", "<BOS_SID>", "<EOS_SID>"]


class GemmaTigerBackbone(nn.Module):
    """Gemma 3 4B + LoRA + vocab extension.

    Args:
        model_name: HF model id (default google/gemma-3-4b-it)
        codebook_sizes: SID 레벨 codebook 크기 (sid_v2 와 일치)
        lora_r, lora_alpha, lora_dropout: LoRA hparam
        load_in_4bit: 양자화 (bitsandbytes) — 메모리 더 줄이려면
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B",
        codebook_sizes: Tuple[int, ...] = (2048, 1024, 512),
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        # Gemma 4 multimodal: vision/audio 의 ClippableLinear 는 LoRA 미지원 →
        # text language_model 의 nn.Linear 들만 정확히 매칭하는 regex 사용.
        target_modules_regex: Optional[str] = (
            r".*language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(gate_proj|up_proj|down_proj))$"
        ),
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.codebook_sizes = codebook_sizes

        # 1) tokenizer 로드 + vocab 확장
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sid_tokens = make_sid_tokens(codebook_sizes)
        new_tokens = self.sid_tokens + BEHAVIOR_TOKENS + STRUCTURAL_TOKENS
        n_added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.n_added_tokens = n_added

        # 2) backbone 로드 (Gemma 4 multimodal, text-only forward 만 사용)
        # Gemma4ForCausalLM 은 체크포인트 키 매칭 안 됨 → ConditionalGeneration 사용.
        # input_ids 만 주면 vision/audio branch 호출 안 됨.
        # attn_implementation: eager (Gemma 4 SDPA 호환 미확인, 안전)
        load_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="eager")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self.lm = Gemma4ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        # 3) embedding 확장 (새 SID/BEH 토큰 자리 추가)
        # mean_resizing=False: transformers 의 mean+cov 자동 init 비활성
        # (메모리 폭증 회피 + 우리가 _init_new_token_embeddings 에서 자체 init)
        self.lm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self._init_new_token_embeddings()

        # 4) Gradient checkpointing (activation 메모리 ~5× 절감, step 시간 +30%)
        # peft 호환: prepare_model_for_kbit_training 으로 input grad enable
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lm.config.use_cache = False
        if hasattr(self.lm, "enable_input_require_grads"):
            self.lm.enable_input_require_grads()

        # 5) LoRA 적용 (text language_model 의 attention + MLP 만, regex 매칭)
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules_regex,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.lm = get_peft_model(self.lm, lora_config)

        # 5) 새 token embedding 은 LoRA 외 별도로 학습 가능하게
        #    (embed_tokens.weight 의 마지막 n_added 행만 grad on)
        self._make_new_token_embeddings_trainable()

    @torch.no_grad()
    def _init_new_token_embeddings(self):
        """새 special token embedding 을 기존 평균 + 작은 noise 로 init."""
        emb = self.lm.get_input_embeddings()
        weight = emb.weight.data  # (V_total, D)
        n_orig = weight.size(0) - self.n_added_tokens
        mean_orig = weight[:n_orig].mean(dim=0)
        std_orig = weight[:n_orig].std(dim=0).clamp(min=1e-4)
        new = mean_orig + std_orig * 0.02 * torch.randn(
            self.n_added_tokens, weight.size(1), dtype=weight.dtype, device=weight.device
        )
        weight[n_orig:] = new

    def _make_new_token_embeddings_trainable(self):
        """LoRA wrap 후에도 새 token embedding row 들은 학습 가능하게."""
        # embed_tokens 전체를 학습 가능으로 만든 뒤, gradient hook 으로 기존 row 는 0 으로
        emb = self.lm.get_input_embeddings()
        emb.weight.requires_grad = True
        n_orig = emb.weight.size(0) - self.n_added_tokens
        n_added = self.n_added_tokens

        def _mask_grad(grad):
            # 기존 vocab row 의 grad 0 → 새 token row 만 학습
            grad[:n_orig] = 0
            return grad
        emb.weight.register_hook(_mask_grad)

        # output layer (lm_head) 도 동일 — 새 token logits 위해
        try:
            lm_head = self.lm.get_output_embeddings()
            if lm_head is not None and lm_head.weight is not emb.weight:
                lm_head.weight.requires_grad = True
                def _mask_grad_head(grad):
                    grad[:n_orig] = 0
                    return grad
                lm_head.weight.register_hook(_mask_grad_head)
        except Exception:
            pass  # tied weights 면 자동 처리됨

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def sid_token_id(self, level: int, code: int) -> int:
        return self.tokenizer.convert_tokens_to_ids(f"<SID_{level}_{code}>")

    def behavior_token_id(self, behavior: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(f"<BEH_{behavior}>")

    def structural_token_id(self, name: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def trainable_params_count(self) -> dict:
        total, trainable = 0, 0
        for p in self.lm.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }
