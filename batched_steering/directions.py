"""Directions: difference of means (reference-repo semantics), plus same-length controls.

All functions return a tensor of shape (n_layers + 1, d) on CPU in fp32, indexed by layer number like hidden_states
(index 0 = embeddings, index L = output of block L).
"""
from __future__ import annotations

import random

import torch

from .model import chat, encode

FEEL = "Tell me how you feel."


def _hidden(lm, texts: list[str], batch: int = 32):
    """Hidden states for each text at fixed shapes. Returns list of (n_layers+1, T_i, d) fp32 CPU tensors (unpadded)."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids, mask, pos = encode(lm.tok, chunk, lm.device)
        hs = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
        hs = torch.stack([h.float() for h in hs], 1).cpu()        # [b, L+1, T, d]
        lens = mask.sum(-1).cpu().tolist()
        T = ids.shape[1]
        for j, n in enumerate(lens):
            out.append(hs[j, :, T - n:])
    return out


def _ids(lm, text):
    return lm.tok(text, add_special_tokens=False)["input_ids"]


def first_difference(lm, prompt, reply, other) -> int:
    a, b = _ids(lm, chat(lm.tok, prompt, reply)), _ids(lm, chat(lm.tok, prompt, other))
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) - 1)


def reply_acts(lm, prompt: str, replies: list[str], others: list[str] | None = None) -> torch.Tensor:
    """Residual stream after every block averaged over each reply's tokens, starting where it differs from its pair
    (or at the first reply token if no pair). -> (N, L+1, d)"""
    starts = [len(_ids(lm, chat(lm.tok, prompt))) if others is None else first_difference(lm, prompt, r, others[j])
              for j, r in enumerate(replies)]
    hs = _hidden(lm, [chat(lm.tok, prompt, r) for r in replies])
    return torch.stack([h[:, s:].mean(1) for h, s in zip(hs, starts)])


def sad_acts(lm, pairs=None, prompt: str = FEEL):
    """(happy_acts, sad_acts), each (N, L+1, d)."""
    if pairs is None:
        from .data.mood_pairs import PAIRS as pairs
    H, S = [h for h, s in pairs], [s for h, s in pairs]
    return reply_acts(lm, prompt, H, S), reply_acts(lm, prompt, S, H)


def sad_direction(lm, pairs=None) -> torch.Tensor:
    happy, sad = sad_acts(lm, pairs)
    return sad.mean(0) - happy.mean(0)


def shuffled_direction(happy: torch.Tensor, sad: torch.Tensor) -> torch.Tensor:
    """Control: flip each pair's label with p=0.5 (same RNG as the reference repo)."""
    flip = [random.Random(i).random() < 0.5 for i in range(len(happy))]
    return torch.stack([(happy[i] - sad[i]) if f else (sad[i] - happy[i]) for i, f in enumerate(flip)]).mean(0)


def random_like(vec: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random direction with the same length as vec (per row if vec is (L+1, d))."""
    g = torch.randn(vec.shape, generator=torch.Generator().manual_seed(seed))
    return g / g.norm(dim=-1, keepdim=True) * vec.norm(dim=-1, keepdim=True)


def match_norm(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True) * ref.norm(dim=-1, keepdim=True)


def mean_acts(lm, texts: list[str]) -> torch.Tensor:
    """Mean over all tokens of each raw text (no chat template), then over texts. -> (L+1, d)"""
    return torch.stack([h.mean(1) for h in _hidden(lm, texts)]).mean(0)


def tomato_direction(lm) -> torch.Tensor:
    from .data.tomato_sentences import OTHERS, TARGET, TEMPLATES
    fill = lambda w: [t.format(w) for t in TEMPLATES]
    return mean_acts(lm, fill(TARGET)) - torch.stack([mean_acts(lm, fill(w)) for w in OTHERS]).mean(0)


def prompt_last_acts(lm, prompts: list[str]) -> torch.Tensor:
    """Residual stream at the last token of each chat-formatted prompt (the end of the assistant header, where the
    model decides how to answer). -> (N, L+1, d)"""
    return torch.stack([h[:, -1] for h in _hidden(lm, [chat(lm.tok, p) for p in prompts])])


def refusal_acts(lm, harmful=None, harmless=None):
    """(harmful_acts, harmless_acts) at the last prompt token, each (N, L+1, d). Default: the train halves."""
    from .data.refusal_prompts import HARMFUL_TRAIN, HARMLESS_TRAIN
    return prompt_last_acts(lm, harmful or HARMFUL_TRAIN), prompt_last_acts(lm, harmless or HARMLESS_TRAIN)


def refusal_direction(lm) -> torch.Tensor:
    """Harmful mean - harmless mean at the last prompt token: (L+1, d). Adding it should make the model refuse."""
    bad, ok = refusal_acts(lm)
    return bad.mean(0) - ok.mean(0)
