import sys, torch
sys.path.insert(0, ".")
from batched_steering import load, Config, Sweep
from batched_steering.loop import Loop
from batched_steering.directions import sad_direction
from batched_steering.data.topics import TEST, prompts
lm = load("Qwen/Qwen3-0.6B", dtype=torch.bfloat16); d = sad_direction(lm); L = 12
P = prompts(TEST)[:6]
cfgs = [Config("x+2", vec=d[L], strength=2.0, layers=(L,)), Config("x+0", vec=d[L], strength=0.0, layers=(L,))]
loopA = Loop(lm, 60, 0.0).generate(cfgs, P)
sw = Sweep(lm, 24, 60, 0.0); sweep24 = sw.generate(cfgs, P); sw.close()
loopB = Loop(lm, 60, 0.0).generate(cfgs, P)
for ci, c in enumerate(cfgs):
    for pi in range(3):
        print(f"{c.name} | {P[pi][:30]}\n   loop-before: {loopA[ci][pi][:80]!r}\n   sweep      : {sweep24[ci][pi][:80]!r}\n   loop-after : {loopB[ci][pi][:80]!r}", flush=True)
