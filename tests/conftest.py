"""Tiny random Qwen3 with the real Qwen3 tokenizer: fast CPU tests of the steering machinery (fp32, greedy)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def lm():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from batched_steering.model import load

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).eval()
    return load(device=torch.device("cpu"), model=model, tok=tok)
