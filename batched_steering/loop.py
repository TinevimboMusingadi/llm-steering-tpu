"""Mode A: one config at a time with plain per-setting hooks (the reference repo's way). Baseline for speed and correctness."""
from __future__ import annotations

import torch

from .generate import Generator, decode
from .model import chat, encode
from .steer import SingleSteer
from .sweep import Config


def apply(s: SingleSteer, c: Config):
    for p in (c, *c.extra):                                   # pushes first, then the removal sees the pushed state
        if p.vec is not None and p.strength != 0:
            for L in p.layers:
                s.add(L, p.strength * p.vec.float(), gen_only=p.gen_only)
    if c.unit is not None and c.ablate != 0:
        s.remove_dir(c.unit.float(), c.ablate, list(c.layers))


class Loop:
    def __init__(self, lm, n_new: int = 60, temperature: float = 0.5):
        self.lm, self.N, self.temperature = lm, n_new, temperature
        self._gens = {}

    def generate(self, configs: list[Config], prompts: list[str], chat_format: bool = True, on_config=None):
        texts = [chat(self.lm.tok, p) if chat_format else p for p in prompts]
        ids, mask, pos = encode(self.lm.tok, texts, self.lm.device)
        key = tuple(ids.shape)
        if key not in self._gens:
            self._gens[key] = Generator(self.lm, ids.shape[0], ids.shape[1], self.N)
        gen = self._gens[key]
        out = []
        for i, c in enumerate(configs):
            with SingleSteer(self.lm.layers) as s:
                apply(s, c)
                out.append(decode(self.lm.tok, gen(ids, mask, pos, self.temperature)))
            if on_config:
                on_config(i)
        return out

    @torch.no_grad()
    def fact_damage(self, configs, prompts, answer_ids, before):
        ids, mask, pos = encode(self.lm.tok, prompts, self.lm.device)
        before = before.to(self.lm.device)
        ans = torch.tensor(answer_ids, device=self.lm.device)
        accs, kls = [], []
        for c in configs:
            with SingleSteer(self.lm.layers) as s:
                apply(s, c)
                lp = self.lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1]
                lp = lp.float().log_softmax(-1)
            accs.append(float((lp.argmax(-1) == ans).float().mean()))
            kls.append(float((before.exp() * (before - lp)).sum(-1).mean()))
        return torch.tensor(accs), torch.tensor(kls)
