"""Mixed batch in bf16: half the rows push 2x sad at L12, half push nothing. Does row position or mixing break it?"""
import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, chat, encode, decode, Generator
from batched_steering.steer import BatchedSteer
from batched_steering.directions import sad_direction
dt = torch.bfloat16 if sys.argv[1] == "bf16" else torch.float32
lm = load("Qwen/Qwen3-0.6B", dtype=dt)
d = sad_direction(lm); B, L = 256, 12
ids, mask, pos = encode(lm.tok, [chat(lm.tok, "Write two sentences about winter.")] * B, lm.device)
gen = Generator(lm, B, ids.shape[1], 24)
st = BatchedSteer(lm.layers, B, lm.d_model, lm.device)
lmask = torch.zeros(B, lm.n_layers); lmask[:, L - 1] = 1
def run(strengths):
    st.load(vec=d[L].expand(B, -1), strength=strengths, layer_mask=lmask, prompt_scale=torch.ones(B)); st.set_token_mask(mask)
    return gen(ids, mask, pos, temperature=0).cpu()
ref_push = run(torch.full((B,), 2.0))[0]; ref_none = run(torch.zeros(B))[0]
for name, s in [("first half pushed", torch.cat([torch.full((128,), 2.0), torch.zeros(128)])),
                ("second half pushed", torch.cat([torch.zeros(128), torch.full((128,), 2.0)])),
                ("alternating", torch.tensor([2.0, 0.0] * 128)),
                ("strengths 0..4 ramp", torch.linspace(0, 4, B))]:
    t = run(s)
    pushed = s > 0
    ok_push = (t[pushed & (s == 2.0)] == ref_push).all(-1) if (s == 2.0).any() else torch.tensor([])
    ok_none = (t[~pushed] == ref_none).all(-1)
    badp = (s == 2.0).nonzero().flatten()[~ok_push].tolist() if len(ok_push) else []
    badn = (~pushed).nonzero().flatten()[~ok_none].tolist()
    print(f"[{sys.argv[1]}] {name:20}: pushed rows wrong {len(badp)} {badp[:3]} | unpushed rows wrong {len(badn)} {badn[:3]}", flush=True)
    if name.startswith("strengths"):
        for r in (0, 64, 127, 128, 200, 255): print(f"     row {r:3} s={s[r]:.2f}: {decode(lm.tok, t[r:r+1])[0][:70]!r}")
