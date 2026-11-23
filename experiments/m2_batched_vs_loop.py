"""M2: batched steering == one-at-a-time steering (greedy), and it is faster.

Runs the M1 configs (sad x-3..+4 at layer 12 + 2 controls) both ways with temperature 0 and compares replies row by row.
Done when: replies match (exactly in fp32; nearly in bf16) and batched wall time < loop wall time.
"""
from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import match_norm, random_like, sad_acts, shuffled_direction
from batched_steering.io import r3, save_json
from batched_steering.loop import Loop
from batched_steering.data.topics import TEST, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--layer", type=int, default=12)
p.add_argument("--warmup", type=int, default=1, help="untimed warm-up passes per mode (compiles)")
args = p.parse_args()
args.temperature = 0.0
lm = setup(args)
L = args.layer

happy, sad = sad_acts(lm)
d = sad.mean(0) - happy.mean(0)
push2 = 2 * d[L]
configs = [Config(f"sad x{s:+d}", vec=d[L], strength=float(s), layers=(L,)) for s in [-3, -2, -1, 0, 1, 2, 3, 4]]
configs += [Config("random", vec=random_like(push2), strength=1.0, layers=(L,)),
            Config("shuffled", vec=match_norm(shuffled_direction(happy, sad)[L], push2), strength=1.0, layers=(L,)),
            Config("remove all", unit=d[L], ablate=1.0, layers=tuple(range(1, lm.n_layers + 1)))]
test = limit(make_prompts(TEST), args.topics)

loop = Loop(lm, args.tokens, 0.0)
sweep = Sweep(lm, args.batch, args.tokens, 0.0)
for _ in range(args.warmup):
    loop.generate(configs[:1], test)
    sweep.generate(configs[:1], test)

with Timer(lm.device) as t_loop:
    a = loop.generate(configs, test)
with Timer(lm.device) as t_batch:
    b = sweep.generate(configs, test)
sweep.close()

rows = [(c.name, i, x == y) for c, xs, ys in zip(configs, a, b) for i, (x, y) in enumerate(zip(xs, ys))]
match = sum(m for *_, m in rows) / len(rows)
mismatches = [{"config": n, "prompt": test[i], "loop": a[ci][i], "batched": b[ci][i]}
              for ci, (c, xs, ys) in enumerate(zip(configs, a, b)) for i, (n, x, y) in enumerate(zip([c.name] * len(xs), xs, ys)) if x != y]
per_config = {c.name: r3(sum(x == y for x, y in zip(xs, ys)) / len(xs)) for c, xs, ys in zip(configs, a, b)}
first_diff = {}
for c, xs, ys in zip(configs, a, b):
    for x, y in zip(xs, ys):
        if x != y:   # how far into the reply the two versions agree (characters)
            first_diff.setdefault(c.name, []).append(next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))))
res = {"rows": len(rows), "match_rate": r3(match), "per_config_match": per_config,
       "mean_chars_before_divergence": {k: r3(sum(v) / len(v)) for k, v in first_diff.items()}, "loop_s": r3(t_loop.seconds), "batched_s": r3(t_batch.seconds),
       "speedup": r3(t_loop.seconds / max(1e-9, t_batch.seconds)), "mismatches": mismatches[:10]}
print("[m2]", {k: v for k, v in res.items() if k != "mismatches"})
save_json("m2_batched_vs_loop", run_name(args, f"m2_{lm.device.type}_{args.dtype}_{args.eager}"), {"args": vars(args), **res})
if args.dtype == "float32":
    assert match == 1.0, f"batched != loop in fp32 ({match:.3f}); first mismatch: {mismatches[:1]}"
print("[m2] PASS" if match == 1.0 else f"[m2] match {match:.3f} (bf16 rounding expected)")
