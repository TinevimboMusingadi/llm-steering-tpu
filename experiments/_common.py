"""Shared CLI for experiment scripts. Every script runs the same on CPU (smoke) and TPU (real)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402

from batched_steering import load, set_eager_mode  # noqa: E402


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--device", default=None, help="tpu | cpu (default: tpu if torch_tpu is installed)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--eager", default="strict", choices=["strict", "defer"], help="TorchTPU eager mode")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--tokens", type=int, default=60, help="new tokens per reply")
    p.add_argument("--batch", type=int, default=256, help="rows per batch (fixed)")
    p.add_argument("--topics", type=int, default=0, help="limit number of topics (0 = all) for smoke runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for result file names")
    p.add_argument("--tiny", action="store_true",
                   help="plumbing smoke test: random 28-layer tiny Qwen3 + keyword mood scorer (no big downloads)")
    return p


def tiny_lm(device):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    return load(device=device, model=Qwen3ForCausalLM(cfg), tok=tok)


def setup(args):
    torch.manual_seed(args.seed)
    eager = set_eager_mode(args.eager)
    if args.tiny:
        os.environ["STEER_FAKE_MOOD"] = "1"
        lm = tiny_lm(args.device)
    else:
        lm = load(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    print(f"[setup] device={lm.device} dtype={lm.dtype} eager={eager} layers={lm.n_layers} d={lm.d_model}", flush=True)
    return lm


def limit(xs, n):
    return xs[:n] if n else xs


def run_name(args, base: str) -> str:
    return f"{base}_{args.tag}" if args.tag else base
