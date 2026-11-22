"""Load Qwen3 on TPU or CPU, chat formatting, and fixed-shape (bucketed, left-padded) tokenization."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import get_device

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
BUCKETS = (32, 64, 128, 256)


@dataclass
class LM:
    tok: object
    model: torch.nn.Module
    device: torch.device

    @property
    def layers(self):
        return self.model.model.layers  # Qwen3-0.6B: 28 blocks, width 1024

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype


def load(model_id: str = DEFAULT_MODEL, device=None, dtype=torch.float32, model=None, tok=None) -> LM:
    """fp32 by default, like the reference repo (bf16 for speed runs). Pass `model`/`tok` to wrap existing ones (tests)."""
    torch.set_grad_enabled(False)
    device = get_device(device) if not isinstance(device, torch.device) else device
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tok.padding_side = "left"
    if model is None:
        if device.type == "tpu":
            # `.to("tpu")` was reported to leave weights on the host; device_map places them properly.
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="tpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    return LM(tok=tok, model=model, device=device)


def chat(tok, prompt: str, reply: str | None = None) -> str:
    """Chat-formatted text for one user prompt (thinking off), optionally followed by an assistant reply."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)
    return text if reply is None else text + reply


def bucket_len(n: int, buckets=BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"sequence of {n} tokens is longer than the largest bucket {buckets[-1]}")


def encode(tok, texts: list[str], device, length: int | None = None, buckets=BUCKETS):
    """Left-pad to a fixed bucket length so shapes repeat. Returns input_ids, attention_mask, position_ids (all [B, T])."""
    raw = tok(texts, add_special_tokens=False)["input_ids"]
    T = length or bucket_len(max(len(r) for r in raw), buckets)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(raw), T), pad, dtype=torch.long)
    mask = torch.zeros((len(raw), T), dtype=torch.long)
    for i, r in enumerate(raw):
        if len(r) > T:
            raise ValueError(f"text {i} has {len(r)} tokens > length {T}")
        if r:
            ids[i, T - len(r):] = torch.tensor(r)
            mask[i, T - len(r):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)
