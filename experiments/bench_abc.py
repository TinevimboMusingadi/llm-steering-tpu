"""Benchmark: mode A (loop, one config at a time) vs B (batched, Strict eager) vs C (batched, DeferAndFuse).

Workload: E1-style sad pushes (random layer x strength) on the 12 tune topics, 60 new tokens, temperature 0.5.
Run once per eager mode (the mode is process-wide):
    python experiments/bench_abc.py --eager strict      # A and B
    python experiments/bench_abc.py --eager defer       # A and C
    python experiments/bench_abc.py --compile           # A and D (decode step under torch.compile, backend "tpu")
Reports wall seconds and configs/s for several sweep sizes, after an untimed warm-up of each shape.
"""
import random

from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import sad_direction
from batched_steering.io import r3, save_json
from batched_steering.loop import Loop
from batched_steering.data.topics import TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--sizes", default="8,32,64,128,224", help="number of configs per batched sweep")
p.add_argument("--loop-configs", type=int, default=8, help="configs timed in the loop (it is slow)")
p.add_argument("--compile", action="store_true", help="mode D: torch.compile the decode step")
p.add_argument("--backend", default=None, help="compile backend (default: tpu on TPU, eager elsewhere)")
args = p.parse_args()
lm = setup(args)

prompts = make_prompts(limit(TUNE, args.topics))
d = sad_direction(lm)
rng = random.Random(args.seed)
configs = [Config(vec=d[L], strength=s, layers=(L,)) for L, s in
           ((rng.randrange(1, lm.n_layers + 1), rng.choice([-2, -1, 0.5, 1, 1.5, 2, 3, 4])) for _ in range(max(int(x) for x in args.sizes.split(","))))]

res = {"args": vars(args), "prompts": len(prompts), "batch_rows": args.batch}
loop = Loop(lm, args.tokens, args.temperature)
loop.generate(configs[:1], prompts)                                    # warm-up
with Timer(lm.device) as t:
    loop.generate(configs[:args.loop_configs], prompts)
res["A_loop"] = {"configs": args.loop_configs, "seconds": r3(t.seconds), "configs_per_s": r3(args.loop_configs / t.seconds)}
print("[bench] A loop:", res["A_loop"], flush=True)

backend = (args.backend or ("tpu" if lm.device.type == "tpu" else "eager")) if args.compile else None
mode = "D_batched_compiled" if args.compile else ("B_batched_strict" if args.eager == "strict" else "C_batched_defer")


def graphs():
    from torch._dynamo.utils import counters
    return {"unique_graphs": int(counters["stats"]["unique_graphs"]), "frames_ok": int(counters["frames"]["ok"])}


sweep = Sweep(lm, args.batch, args.tokens, args.temperature, compile_backend=backend)
with Timer(lm.device) as tw:
    sweep.generate(configs[:1], prompts)                               # warm-up (compiles the shapes)
res["warmup"] = {"seconds": r3(tw.seconds), "backend": backend, **graphs()}
print("[bench] warm-up:", res["warmup"], flush=True)
res[mode] = []
for n in (int(x) for x in args.sizes.split(",")):
    g0 = graphs()
    with Timer(lm.device) as t:
        sweep.generate(configs[:n], prompts)
    g1 = graphs()
    row = {"configs": n, "rows": n * len(prompts), "seconds": r3(t.seconds), "configs_per_s": r3(n / t.seconds),
           "speedup_vs_loop": r3((n / t.seconds) / res["A_loop"]["configs_per_s"]),
           "new_graphs": g1["unique_graphs"] - g0["unique_graphs"]}
    res[mode].append(row)
    print(f"[bench] {mode}:", row, flush=True)
sweep.close()
if args.compile:
    res["zero_recompiles_after_warmup"] = all(r["new_graphs"] == 0 for r in res[mode])
    print("[bench] zero new graphs after warm-up:", res["zero_recompiles_after_warmup"], flush=True)
save_json("bench_abc", run_name(args, f"bench_{lm.device.type}_{args.dtype}_{'compile' if args.compile else args.eager}"), res)
print("[bench] done")
