"""All 256 rows get the SAME steering config and prompt -> all rows must be identical. Which hook math breaks in bf16?"""
import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, chat, encode, decode, Generator
from batched_steering.steer import BatchedSteer, _split
from batched_steering.directions import sad_direction
lm = load("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
d = sad_direction(lm)
B, L = 256, 12

def hook_native(self, i):   # variant: do all math in the activation dtype (bf16)
    def hook(module, inputs, out):
        h, pack = _split(out); c = self.cfg; dt = h.dtype
        m = c["layer_mask"][:, i].to(dt)
        pos = (c["prompt_scale"].to(dt)[:, None] * self.token_mask.to(dt)) if h.shape[1] > 1 else torch.ones_like(m)[:, None]
        u = c["unit"].to(dt)[:, None, :]
        coef = (h * u).sum(-1)
        delta = ((c["strength"].to(dt) * m)[:, None] * pos)[..., None] * c["vec"].to(dt)[:, None, :] \
                - (((c["ablate"].to(dt) * m)[:, None] * pos) * coef)[..., None] * u
        return pack(h + delta)
    return hook

ids, mask, pos = encode(lm.tok, [chat(lm.tok, "Write two sentences about winter.")] * B, lm.device)
gen = Generator(lm, B, ids.shape[1], 24)
for name in ("current_fp32_math", "native_bf16_math"):
    if name == "native_bf16_math":
        BatchedSteer._hook = hook_native
    st = BatchedSteer(lm.layers, B, lm.d_model, lm.device)
    lmask = torch.zeros(B, lm.n_layers); lmask[:, L - 1] = 1
    st.load(vec=d[L].expand(B, -1), strength=torch.full((B,), 2.0), layer_mask=lmask, prompt_scale=torch.ones(B))
    st.set_token_mask(mask)
    toks = gen(ids, mask, pos, temperature=0).cpu(); st.remove()
    bad = (~(toks == toks[0]).all(-1)).nonzero().flatten().tolist()
    print(f"{name}: rows differing from row 0: {len(bad)} range={bad[:1]}..{bad[-1:]}", flush=True)
    print("   row0  :", decode(lm.tok, toks[:1])[0][:80])
    if bad: print("   row", bad[0], ":", decode(lm.tok, toks[bad[0]:bad[0]+1])[0][:80])
