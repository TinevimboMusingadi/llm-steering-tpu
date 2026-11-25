"""TPU check for the stretch round (greedy, fp32, Qwen3-0.6B):
  1. mixed rows (two pushes, push + removal) batched == one at a time (loop)
  2. mode D: compiled decode step gives the same tokens as eager
python scripts/check_mix_compile.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from batched_steering import Config, Sweep, load  # noqa: E402
from batched_steering.directions import refusal_direction, sad_direction, tomato_direction  # noqa: E402
from batched_steering.loop import Loop  # noqa: E402
from batched_steering.data.topics import TEST, prompts as make_prompts  # noqa: E402

lm = load("Qwen/Qwen3-0.6B")
N = lm.n_layers
sad, tom, ref = sad_direction(lm), tomato_direction(lm), refusal_direction(lm)
P = make_prompts(TEST[:6])
configs = [
    Config("sad L12 x2 + tomato L18 x10", vec=sad[12], strength=2.0, layers=(12,),
           extra=[Config(vec=tom[18], strength=10.0, layers=(18,), gen_only=True)]),
    Config("sad L17 x1.5 + tomato L17-18 x12.3", vec=sad[17], strength=1.5, layers=(17,),
           extra=[Config(vec=tom[17] / 2, strength=12.3, layers=(17, 18), gen_only=True)]),
    Config("sad L12 x2 + remove", vec=sad[12], strength=2.0, unit=sad[12], ablate=1.0, layers=(12,)),
    Config("remove refusal L25 x0.5", unit=ref[25], ablate=0.5, layers=tuple(range(1, N + 1))),
    Config("induce refusal L20 x2", vec=ref[20], strength=2.0, layers=(20,)),
]
sw = Sweep(lm, 64, 40, temperature=0)
batched = sw.generate(configs, P)
sw.close()
loop = Loop(lm, 40, temperature=0).generate(configs, P)
ok = 0
for c, b, l in zip(configs, batched, loop):
    same = sum(x == y for x, y in zip(b, l))
    ok += same
    print(f"[check] mix  {c.name:38s} batched == loop: {same}/{len(P)}")
print(f"[check] mix total {ok}/{len(P) * len(configs)}")

sw_e = Sweep(lm, 64, 40, temperature=0)
sw_c = Sweep(lm, 64, 40, temperature=0, compile_backend="tpu")
e = sw_e.generate(configs, P)
c = sw_c.generate(configs, P)
sw_e.close(); sw_c.close()
same = sum(x == y for a, b in zip(e, c) for x, y in zip(a, b))
print(f"[check] compile == eager: {same}/{len(P) * len(configs)}")


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


for tag, A, B in (("mix", batched, loop), ("compile", e, c)):
    for cf, ra, rb in zip(configs, A, B):
        for x, y in zip(ra, rb):
            if x != y:
                print(f"[check] {tag} differs: {cf.name}: same for the first {first_diff(x, y)} of {len(x)} chars")
