"""M2's exact configs through Sweep in bf16: batch 256 vs batch 24 (one config per chunk). Which rows differ?"""
import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, Config, Sweep
from batched_steering.steer import BatchedSteer, _split
from batched_steering.directions import sad_acts, random_like, match_norm, shuffled_direction
from batched_steering.data.topics import TEST, prompts
lm = load("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
happy, sad = sad_acts(lm); d = sad.mean(0) - happy.mean(0); L = 12; push2 = 2 * d[L]
configs = [Config(f"sad x{s:+d}", vec=d[L], strength=float(s), layers=(L,)) for s in [-3, -2, -1, 0, 1, 2, 3, 4]]
configs += [Config("random", vec=random_like(push2), strength=1.0, layers=(L,)),
            Config("shuffled", vec=match_norm(shuffled_direction(happy, sad)[L], push2), strength=1.0, layers=(L,)),
            Config("remove all", unit=d[L], ablate=1.0, layers=tuple(range(1, 29)))]
P = prompts(TEST)
def run(B):
    sw = Sweep(lm, B, 40, 0.0); out = sw.generate(configs, P); sw.close(); return out
small, big = run(24), run(256)
rowbad = [ci * 24 + pi for ci in range(len(configs)) for pi in range(24) if small[ci][pi] != big[ci][pi]]
print("rows differing (global row index):", len(rowbad), "min", min(rowbad, default=None), "max", max(rowbad, default=None), flush=True)
for ci, c in enumerate(configs):
    n = sum(small[ci][pi] != big[ci][pi] for pi in range(24)); print(f"   {c.name:12} differ {n}/24")
r = rowbad[0] if rowbad else None
if r is not None:
    ci, pi = divmod(r, 24); print("example", configs[ci].name, P[pi], "\n  B=24 :", small[ci][pi][:90], "\n  B=256:", big[ci][pi][:90])
