"""Steering hooks.

BatchedSteer: one forward hook on EVERY block. Each batch row b carries its own experiment:
    vec[b]        [d]  direction to add (raw, not normalised: strength 1 = add it once)
    strength[b]        how hard to push (0 = no push)
    unit[b]       [d]  unit direction to project out
    ablate[b]          how much of it to remove (0 = none, 1 = fully)
    layer_mask[b] [L]  1.0 on the blocks to edit, 0 elsewhere
    prompt_scale[b]    1 = also edit prompt tokens, 0 = only generated tokens
    vec2, strength2, layer_mask2, prompt_scale2: a second push with its own blocks and positions (mixing two
                       directions, e.g. sad on every token at L12 + tomato on generated tokens at L18). Zero = off.
    token_mask    [B,T] 1 on real prompt tokens, 0 on left padding (prefill only)

The removal sees the pushed state (h + pushes), the same order as SingleSteer's hooks (pushes registered first).
No Python branching per row, and all buffers are allocated once and updated in place with .copy_(), so the
compiler sees the same tensors every sweep. Prefill vs decode is told apart by the sequence length (static).

Layers are numbered like hidden_states / the reference repo: layer L (1..n) = output of model.model.layers[L-1].
"""
from __future__ import annotations

import torch


def _split(out):
    if isinstance(out, tuple):
        return out[0], lambda h: (h,) + tuple(out[1:])
    return out, lambda h: h


class BatchedSteer:
    KEYS = ("vec", "unit", "strength", "ablate", "layer_mask", "prompt_scale",
            "vec2", "strength2", "layer_mask2", "prompt_scale2")

    def __init__(self, layers, batch_size: int, d_model: int, device, dtype=torch.float32):
        self.layers, self.B, self.d, self.device = layers, batch_size, d_model, device
        L = len(layers)
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        self.cfg = dict(vec=z(batch_size, d_model), unit=z(batch_size, d_model), strength=z(batch_size),
                        ablate=z(batch_size), layer_mask=z(batch_size, L), prompt_scale=z(batch_size),
                        vec2=z(batch_size, d_model), strength2=z(batch_size), layer_mask2=z(batch_size, L),
                        prompt_scale2=z(batch_size))
        self.token_mask = None  # [B, T_prompt], set per prompt bucket
        self.enabled = True
        self.handles = [blk.register_forward_hook(self._hook(i)) for i, blk in enumerate(layers)]

    # ---- loading a sweep chunk (same shapes every time) ----
    def load(self, **new):
        for k, v in new.items():
            if k == "token_mask":
                self.set_token_mask(v)
                continue
            buf = self.cfg[k]
            v = torch.as_tensor(v, dtype=buf.dtype)
            if v.shape != buf.shape:
                raise ValueError(f"{k}: expected shape {tuple(buf.shape)}, got {tuple(v.shape)}")
            buf.copy_(v.to(self.device))

    def set_token_mask(self, attention_mask: torch.Tensor):
        m = attention_mask.to(self.device, self.cfg["vec"].dtype)
        if self.token_mask is None or self.token_mask.shape != m.shape:
            self.token_mask = m.clone()  # new prompt bucket -> new buffer (one recompile per bucket)
        else:
            self.token_mask.copy_(m)

    def clear(self):
        for v in self.cfg.values():
            v.zero_()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    # ---- the hook ----
    def _hook(self, i: int):
        def hook(module, inputs, out):
            if not self.enabled:
                return out
            h, pack = _split(out)                                          # [B, T, d]
            c = self.cfg
            m, m2 = c["layer_mask"][:, i], c["layer_mask2"][:, i]           # [B]
            T = h.shape[1]
            if T > 1:                                                      # prefill: prompt tokens
                tm = self.token_mask if self.token_mask is not None and self.token_mask.shape[1] == T                     else torch.ones(h.shape[:2], device=h.device, dtype=m.dtype)
                pos = c["prompt_scale"][:, None] * tm                      # [B, T]
                pos2 = c["prompt_scale2"][:, None] * tm
            else:                                                          # decode: one generated token
                pos = pos2 = torch.ones_like(m)[:, None]                   # [B, 1]
            w_push = (c["strength"] * m)[:, None] * pos                    # [B, T]
            w_push2 = (c["strength2"] * m2)[:, None] * pos2                # [B, T]
            w_abl = (c["ablate"] * m)[:, None] * pos                       # [B, T]
            u = c["unit"][:, None, :]                                      # [B, 1, d]
            hf = h.float() if h.dtype != c["vec"].dtype else h
            push = w_push[..., None] * c["vec"][:, None, :] + w_push2[..., None] * c["vec2"][:, None, :]
            coef = ((hf + push) * u).sum(-1)                               # [B, T]
            delta = push - (w_abl * coef)[..., None] * u
            return pack(h + delta.to(h.dtype))
        return hook


class SingleSteer:
    """The reference repo's semantics, one setting for the whole batch (mode A / correctness baseline).
    push: add `vec` after block `layer` (all positions, or decode-only if gen_only).
    remove: o - s (o.r)r after the blocks in `which` (default all)."""

    def __init__(self, layers):
        self.layers, self.handles = layers, []

    def add(self, layer: int, vec: torch.Tensor, gen_only: bool = False, spread: int = 1):
        n = len(self.layers)
        v = (vec / spread)

        def fn(m, i, out):
            h, pack = _split(out)
            if gen_only and h.shape[1] != 1:
                return out
            return pack(h + v.to(h.device, h.dtype))
        for l in range(layer, min(layer + spread, n + 1)):
            self.handles.append(self.layers[l - 1].register_forward_hook(fn))
        return self

    def remove_dir(self, direction: torch.Tensor, strength: float = 1.0, which=None):
        r = direction / direction.norm()

        def fn(m, i, out):
            h, pack = _split(out)
            rr = r.to(h.device, h.dtype)
            # elementwise multiply + sum, not h @ rr: on TPU an fp32 matmul runs at bf16 input precision by default,
            # which drifts from the batched hook over 28 blocks (seen in M2: 5/264 rows differed late in the text)
            return pack(h - strength * (h * rr).sum(-1, keepdim=True) * rr)
        for l in (which or range(1, len(self.layers) + 1)):
            self.handles.append(self.layers[l - 1].register_forward_hook(fn))
        return self

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()
        return False
