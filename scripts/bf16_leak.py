"""No steering. Different prompts per row. Does row r at batch 256 match the same prompt run in a small batch?"""
import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, chat, encode, decode, Generator
from batched_steering.data.topics import TEST, prompts
dt = torch.bfloat16 if sys.argv[1] == "bf16" else torch.float32
attn = sys.argv[2] if len(sys.argv) > 2 else None
from transformers import AutoModelForCausalLM
lm = load("Qwen/Qwen3-0.6B", dtype=dt)
if attn: lm.model.set_attn_implementation(attn)
print("attn impl:", lm.model.config._attn_implementation, flush=True)
P = prompts(TEST)                       # 24 different prompts
texts = [chat(lm.tok, p) for p in P]
ids, mask, pos = encode(lm.tok, texts, lm.device, length=32)
small = Generator(lm, 24, 32, 24)(ids, mask, pos, temperature=0).cpu()
for B in (128, 160, 256):
    idx = torch.arange(B) % 24
    big = Generator(lm, B, 32, 24)(ids[idx.to(lm.device)], mask[idx.to(lm.device)], pos[idx.to(lm.device)], temperature=0).cpu()
    bad = [r for r in range(B) if not torch.equal(big[r], small[idx[r]])]
    print(f"[{sys.argv[1]}] B={B}: rows differing from small-batch run: {len(bad)}  first={bad[:4]}", flush=True)
    if bad:
        r = bad[0]; print("    small:", decode(lm.tok, small[idx[r]:idx[r]+1])[0][:70], "\n    big  :", decode(lm.tok, big[r:r+1])[0][:70])
