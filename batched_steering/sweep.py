"""Sweep: a list of steering configs x a list of prompts -> fixed-size batches -> results split back per config.

Rows are laid out config-major: row = config_index * n_prompts + prompt_index. The last chunk is padded with
dummy rows (no push, no removal) so every chunk has exactly `batch_size` rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .generate import Generator, decode
from .model import chat, encode
from .steer import BatchedSteer


@dataclass
class Config:
    name: str = ""
    vec: torch.Tensor | None = None       # [d] raw push direction
    strength: float = 0.0
    unit: torch.Tensor | None = None      # [d] direction to project out (normalised here)
    ablate: float = 0.0
    layers: tuple[int, ...] = ()          # 1-indexed blocks to edit
    gen_only: bool = False                # push only on generated tokens (tomato)
    meta: dict = field(default_factory=dict)


def row_tensors(configs: list[Config | None], n_layers: int, d: int) -> dict[str, torch.Tensor]:
    """Steering buffers for one chunk of rows (None = dummy row)."""
    B = len(configs)
    t = dict(vec=torch.zeros(B, d), unit=torch.zeros(B, d), strength=torch.zeros(B), ablate=torch.zeros(B),
             layer_mask=torch.zeros(B, n_layers), prompt_scale=torch.zeros(B))
    for b, c in enumerate(configs):
        if c is None:
            continue
        if c.vec is not None:
            t["vec"][b] = c.vec.float()
            t["strength"][b] = c.strength
        if c.unit is not None:
            t["unit"][b] = c.unit.float() / c.unit.float().norm()
            t["ablate"][b] = c.ablate
        for L in c.layers:
            t["layer_mask"][b, L - 1] = 1.0
        t["prompt_scale"][b] = 0.0 if c.gen_only else 1.0
    return t


class Sweep:
    """Owns one BatchedSteer (+ Generator per prompt bucket) and runs configs x prompts through them."""

    def __init__(self, lm, batch_size: int = 256, n_new: int = 60, temperature: float = 0.5):
        self.lm, self.B, self.N, self.temperature = lm, batch_size, n_new, temperature
        self.steer = BatchedSteer(lm.layers, batch_size, lm.d_model, lm.device)
        self.steer.enabled = False          # hooks only act inside this Sweep's own calls
        self._gens: dict[int, Generator] = {}
        self.chunks_run = 0

    def close(self):
        self.steer.remove()

    def _gen(self, T: int) -> Generator:
        if T not in self._gens:
            self._gens[T] = Generator(self.lm, self.B, T, self.N)
        return self._gens[T]

    def _chunks(self, configs, n_prompts):
        rows = [(ci, pi) for ci in range(len(configs)) for pi in range(n_prompts)]
        for s in range(0, len(rows), self.B):
            part = rows[s:s + self.B]
            yield part + [None] * (self.B - len(part))

    def generate(self, configs: list[Config], prompts: list[str], chat_format: bool = True, on_chunk=None):
        """-> list (per config) of list (per prompt) of reply strings."""
        texts = [chat(self.lm.tok, p) if chat_format else p for p in prompts]
        ids_all, mask_all, pos_all = encode(self.lm.tok, texts, "cpu")
        gen = self._gen(ids_all.shape[1])
        out = [[None] * len(prompts) for _ in configs]
        self.steer.enabled = True
        try:
            self._run_generate(configs, prompts, ids_all, mask_all, pos_all, gen, out, on_chunk)
        finally:
            self.steer.enabled = False
        return out

    def _run_generate(self, configs, prompts, ids_all, mask_all, pos_all, gen, out, on_chunk):
        for chunk in self._chunks(configs, len(prompts)):
            pidx = torch.tensor([r[1] if r else 0 for r in chunk])
            ids, mask, pos = (x[pidx].to(self.lm.device) for x in (ids_all, mask_all, pos_all))
            self.steer.load(**row_tensors([configs[r[0]] if r else None for r in chunk], self.lm.n_layers, self.lm.d_model))
            self.steer.set_token_mask(mask)
            replies = decode(self.lm.tok, gen(ids, mask, pos, self.temperature))
            for r, text in zip(chunk, replies):
                if r:
                    out[r[0]][r[1]] = text
            self.chunks_run += 1
            if on_chunk:
                on_chunk(self.chunks_run)

    def _last_lp(self, configs, prompts):
        """Yields (chunk, [B, vocab] log-probs on device) for raw prompts under each row's steering."""
        ids_all, mask_all, pos_all = encode(self.lm.tok, prompts, "cpu")
        for chunk in self._chunks(configs, len(prompts)):
            pidx = torch.tensor([r[1] if r else 0 for r in chunk])
            ids, mask, pos = (x[pidx].to(self.lm.device) for x in (ids_all, mask_all, pos_all))
            self.steer.load(**row_tensors([configs[r[0]] if r else None for r in chunk], self.lm.n_layers, self.lm.d_model))
            self.steer.set_token_mask(mask)
            self.steer.enabled = True
            try:
                lp = self.lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1]
            finally:
                self.steer.enabled = False
            yield chunk, pidx, lp.float().log_softmax(-1)

    @torch.no_grad()
    def last_logprobs(self, configs: list[Config], prompts: list[str]) -> torch.Tensor:
        """[n_configs, n_prompts, vocab] on CPU. Only for a few configs (152k vocab!): use fact_damage for sweeps."""
        res = None
        for chunk, _, lp in self._last_lp(configs, prompts):
            lp = lp.cpu()
            if res is None:
                res = torch.zeros(len(configs), len(prompts), lp.shape[-1])
            for r, row in zip(chunk, lp):
                if r:
                    res[r[0], r[1]] = row
        return res

    @torch.no_grad()
    def fact_damage(self, configs: list[Config], prompts: list[str], answer_ids: list[int], before: torch.Tensor):
        """Per config: (share of facts whose argmax is the answer, mean KL(before || after)), computed on device.
        before: [n_prompts, vocab] unsteered log-probs."""
        before = before.to(self.lm.device)
        ans = torch.tensor(answer_ids, device=self.lm.device)
        correct = torch.zeros(len(configs), len(prompts))
        kls = torch.zeros(len(configs), len(prompts))
        for chunk, pidx, lp in self._last_lp(configs, prompts):
            pidx = pidx.to(self.lm.device)
            b = before[pidx]
            kl = (b.exp() * (b - lp)).sum(-1).cpu()
            ok = (lp.argmax(-1) == ans[pidx]).float().cpu()
            for r, k, o in zip(chunk, kl, ok):
                if r:
                    kls[r[0], r[1]], correct[r[0], r[1]] = k, o
        return correct.mean(1), kls.mean(1)
