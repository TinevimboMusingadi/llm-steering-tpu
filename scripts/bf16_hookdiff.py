import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, chat, encode
from batched_steering.steer import BatchedSteer, SingleSteer
from batched_steering.directions import sad_direction
from batched_steering.data.topics import TEST, prompts
dt = torch.bfloat16 if sys.argv[1] == "bf16" else torch.float32
lm = load("Qwen/Qwen3-0.6B", dtype=dt); d = sad_direction(lm); L = 12
P = prompts(TEST); ids, mask, pos = encode(lm.tok, [chat(lm.tok, p) for p in P], lm.device); B = len(P)
def outs():
    seen = {}
    hs = [lm.layers[i].register_forward_hook(lambda m, inp, o, i=i: seen.__setitem__(i, o.float().cpu())) for i in (L - 1, L, 27)]
    lg = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1].float().cpu()
    [h.remove() for h in hs]; return seen, lg
base, lg0 = outs()
with SingleSteer(lm.layers) as s:
    s.add(L, 2 * d[L]); single, lg1 = outs()
st = BatchedSteer(lm.layers, B, lm.d_model, lm.device)
lmask = torch.zeros(B, lm.n_layers); lmask[:, L - 1] = 1
st.load(vec=d[L].expand(B, -1), strength=torch.full((B,), 2.0), layer_mask=lmask, prompt_scale=torch.ones(B)); st.set_token_mask(mask)
batched, lg2 = outs(); st.remove()
real = mask.bool().cpu()
for i in (L - 1, L, 27):
    a, b = single[i][real], batched[i][real]
    print(f"[{sys.argv[1]}] block {i+1}: |single-base| {(single[i][real]-base[i][real]).norm(dim=-1).mean():.3f}  |batched-base| {(b-base[i][real]).norm(dim=-1).mean():.3f}  max|single-batched| {(a-b).abs().max():.4f}")
print(f"[{sys.argv[1]}] last-token argmax equal: {(lg1.argmax(-1)==lg2.argmax(-1)).float().mean():.3f}  |  expected push norm {(2*d[L]).norm():.3f}")
