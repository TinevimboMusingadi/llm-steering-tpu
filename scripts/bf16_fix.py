import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, Config, Sweep
from batched_steering.loop import Loop
from batched_steering.steer import BatchedSteer, _split
from batched_steering.directions import sad_acts, random_like, match_norm, shuffled_direction
from batched_steering.data.topics import TEST, prompts
lm = load("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
happy, sad = sad_acts(lm); d = sad.mean(0) - happy.mean(0); L = 12; push2 = 2 * d[L]
allc = [Config(f"sad x{s:+d}", vec=d[L], strength=float(s), layers=(L,)) for s in [-3, -2, -1, 0, 1, 2, 3, 4]]
allc += [Config("random", vec=random_like(push2), strength=1.0, layers=(L,)),
         Config("shuffled", vec=match_norm(shuffled_direction(happy, sad)[L], push2), strength=1.0, layers=(L,))]
P = prompts(TEST)
ref = Loop(lm, 60, 0.0).generate(allc, P)
def score(tag, B):
    sw = Sweep(lm, B, 60, 0.0); b = sw.generate(allc, P); sw.close()
    n = sum(x == y for xs, ys in zip(ref, b) for x, y in zip(xs, ys)); print(f"{tag}: {n}/{len(allc)*24} rows match loop", flush=True)
score("current hook, B=256", 256)
score("current hook, B=128", 128)
orig = BatchedSteer._hook
def native(self, i):
    def hook(module, inputs, out):
        h, pack = _split(out); c = self.cfg; dt = h.dtype
        m = c["layer_mask"][:, i].to(dt)
        pos = (c["prompt_scale"].to(dt)[:, None] * self.token_mask.to(dt)) if h.shape[1] > 1 else torch.ones_like(m)[:, None]
        u = c["unit"].to(dt)[:, None, :]
        delta = ((c["strength"].to(dt) * m)[:, None] * pos)[..., None] * c["vec"].to(dt)[:, None, :] \
                - (((c["ablate"].to(dt) * m)[:, None] * pos) * (h * u).sum(-1))[..., None] * u
        return pack(h + delta)
    return hook
BatchedSteer._hook = native
score("bf16-native hook, B=256", 256)
def fp32_nomask(self, i):   # current math but build the [B,T,d] weight with explicit expand instead of broadcasting
    def hook(module, inputs, out):
        h, pack = _split(out); c = self.cfg
        m = c["layer_mask"][:, i]; T = h.shape[1]
        pos = (c["prompt_scale"][:, None] * self.token_mask) if T > 1 else torch.ones_like(m)[:, None]
        w = ((c["strength"] * m)[:, None] * pos).unsqueeze(-1).expand(-1, -1, h.shape[-1]).contiguous()
        delta = w * c["vec"].unsqueeze(1).expand(-1, T, -1).contiguous()
        u = c["unit"][:, None, :]
        delta = delta - (((c["ablate"] * m)[:, None] * pos) * (h.float() * u).sum(-1))[..., None] * u
        return pack((h.float() + delta).to(h.dtype))
    return hook
BatchedSteer._hook = fp32_nomask
score("fp32 add-then-cast hook, B=256", 256)
