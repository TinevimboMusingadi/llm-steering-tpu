"""Hand-written generation: one prefill (last-position logits only) + a fixed number of 1-token decode steps
on a static KV cache. Fixed shapes end to end: [B, T_bucket] prompt, [B, T_bucket + n_new] mask, [B, 1] steps.
We keep generating after EOS and cut the text afterwards, so every row and every batch runs the same program.
"""
from __future__ import annotations

import torch
from transformers import StaticCache


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Greedy if temperature == 0, else plain temperature sampling (top_k=0, top_p=1 like the reference)
    via the Gumbel-max trick: argmax(logits/t + Gumbel noise). Stays on device, no .item()."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, device=logits.device).clamp_(1e-10, 1.0 - 1e-7)
    return (logits / temperature - torch.log(-torch.log(u))).argmax(-1)


class Generator:
    """Reusable generator for a fixed (batch, prompt length, n_new): owns the static cache and the mask buffer."""

    def __init__(self, lm, batch_size: int, prompt_len: int, n_new: int, compile_backend: str | None = None):
        self.lm, self.B, self.T, self.N = lm, batch_size, prompt_len, n_new
        self.max_len = prompt_len + n_new
        self.cache = StaticCache(config=lm.model.config, max_cache_len=self.max_len)
        self.mask = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=lm.device)
        self.tokens = torch.zeros(batch_size, n_new, dtype=torch.long, device=lm.device)
        # cache positions as device tensors (not Python ints), so a compiled decode step sees the same program every step
        self.cache_pos = torch.arange(self.max_len, device=lm.device)
        self.step = self.decode_step
        if compile_backend:   # mode D: compile only the 1-token decode step (prefill stays eager)
            self.step = torch.compile(self.decode_step, backend=compile_backend, dynamic=False)

    def _forward(self, ids, pos, cache_pos):
        out = self.lm.model(input_ids=ids, attention_mask=self.mask, position_ids=pos, cache_position=cache_pos,
                            past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        return out.logits[:, -1]

    def decode_step(self, nxt, pos, cache_pos):
        """One generated token for every row. Kept as its own method so it can be torch.compile'd (mode D)."""
        return self._forward(nxt[:, None], pos, cache_pos)

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, position_ids, temperature: float = 0.5) -> torch.Tensor:
        B, T = input_ids.shape
        assert (B, T) == (self.B, self.T), f"generator built for {(self.B, self.T)}, got {(B, T)}"
        self.cache.reset()
        self.mask.zero_()
        self.mask[:, :T] = attention_mask
        logits = self._forward(input_ids, position_ids, self.cache_pos[:T])   # prefill
        pos = position_ids[:, -1:] + 1                               # [B, 1]
        for i in range(self.N):
            nxt = sample(logits, temperature)
            self.tokens[:, i] = nxt
            if i == self.N - 1:
                break
            self.mask[:, T + i] = 1
            logits = self.step(nxt, pos, self.cache_pos[T + i:T + i + 1])
            pos = pos + 1
        return self.tokens.clone()


def stop_ids(tok) -> set[int]:
    ids = {tok.eos_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        t = tok.convert_tokens_to_ids(s)
        if isinstance(t, int) and t >= 0:
            ids.add(t)
    return {i for i in ids if i is not None}


def decode(tok, tokens: torch.Tensor) -> list[str]:
    """Cut each row at its first stop token, then detokenize."""
    stops = stop_ids(tok)
    out = []
    for row in tokens.cpu().tolist():
        cut = next((j for j, t in enumerate(row) if t in stops), len(row))
        out.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return out
