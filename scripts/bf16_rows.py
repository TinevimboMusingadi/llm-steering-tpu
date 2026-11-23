"""Is bf16 generation broken for rows >= 128 of a big batch on TPU, with NO steering hooks?"""
import sys, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
from batched_steering import load, Generator, chat, encode, decode
for dtype in (torch.bfloat16, torch.float32):
    lm = load("Qwen/Qwen3-0.6B", dtype=dtype)
    for B in (128, 129, 256):
        ids, mask, pos = encode(lm.tok, [chat(lm.tok, "Write two sentences about winter.")] * B, lm.device)
        toks = Generator(lm, B, ids.shape[1], 24)(ids, mask, pos, temperature=0).cpu()
        same = (toks == toks[0]).all(-1)
        bad = (~same).nonzero().flatten().tolist()
        print(f"{str(dtype):15} B={B}: rows differing from row 0: {len(bad)} first={bad[:3]}", flush=True)
        if bad: print("   row0:", decode(lm.tok, toks[:1])[0][:70], "\n   bad :", decode(lm.tok, toks[bad[0]:bad[0]+1])[0][:70], flush=True)
    del lm
