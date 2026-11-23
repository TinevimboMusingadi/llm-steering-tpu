import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, Config, Sweep
from batched_steering.loop import Loop
from batched_steering.directions import sad_acts, random_like, match_norm, shuffled_direction
from batched_steering.data.topics import TEST, prompts
lm = load("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
happy, sad = sad_acts(lm); d = sad.mean(0) - happy.mean(0); L = 12; push2 = 2 * d[L]
allc = [Config(f"sad x{s:+d}", vec=d[L], strength=float(s), layers=(L,)) for s in [-3, -2, -1, 0, 1, 2, 3, 4]]
allc += [Config("random", vec=random_like(push2), strength=1.0, layers=(L,)),
         Config("shuffled", vec=match_norm(shuffled_direction(happy, sad)[L], push2), strength=1.0, layers=(L,)),
         Config("remove all", unit=d[L], ablate=1.0, layers=tuple(range(1, 29)))]
P = prompts(TEST)
def cmp(tag, configs, B, warm=False):
    loop, sw = Loop(lm, 60, 0.0), Sweep(lm, B, 60, 0.0)
    if warm: loop.generate(configs[:1], P); sw.generate(configs[:1], P)
    a = loop.generate(configs, P); b = sw.generate(configs, P); sw.close()
    per = {c.name: sum(x == y for x, y in zip(xs, ys)) for c, xs, ys in zip(configs, a, b)}
    print(f"{tag}: {per}", flush=True)
    i = next((ci for ci, c in enumerate(configs) if per[c.name] < 24), None)
    if i is not None:
        pi = next(p for p in range(24) if a[i][p] != b[i][p]); print("   loop :", a[i][pi][:80], "\n   sweep:", b[i][pi][:80], flush=True)
cmp("A: x+2 alone, B=256", [allc[5]], 256)
cmp("B: all but remove, B=256", allc[:10], 256)
cmp("C: all, B=256, no warmup", allc, 256)
cmp("D: all, B=256, warmup (=M2)", allc, 256, warm=True)
