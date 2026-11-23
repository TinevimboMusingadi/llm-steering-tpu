"""E1: where does the sad direction work? Full grid in batched sweeps.

layers 1..28 x strengths x 3 directions (sad, random same-length, shuffled labels) = 672 configs by default,
each on the tune topics. Measures mood per config and fact damage (accuracy + KL) per config. Checkpoints after
every direction so a dropped Colab runtime loses little.
"""
from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import match_norm, random_like, sad_acts, shuffled_direction
from batched_steering.io import append_samples, load_json, r3, save_json
from batched_steering.score import avg, facts, mood
from batched_steering.data.topics import TEST, TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--strengths", default="-2,-1,0.5,1,1.5,2,3,4")
p.add_argument("--topic-set", default="tune", choices=["tune", "test"])
p.add_argument("--layers", default="", help="comma list (default: all)")
args = p.parse_args()
lm = setup(args)

strengths = [float(s) for s in args.strengths.split(",")]
layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(1, lm.n_layers + 1))
topics = limit(TUNE if args.topic_set == "tune" else TEST, args.topics)
prompts = make_prompts(topics)
name = run_name(args, f"e1_{lm.device.type}_{args.dtype}_{args.eager}")

happy, sad = sad_acts(lm)
d = sad.mean(0) - happy.mean(0)                                         # (L+1, d)
dirs = {"sad": d, "random": random_like(d, seed=0), "shuffled": match_norm(shuffled_direction(happy, sad), d)}

fp, fa = facts(lm.tok)
sweep = Sweep(lm, args.batch, args.tokens, args.temperature)
before = sweep.last_logprobs([Config()], fp)[0]
base_acc = float((before.argmax(-1) == __import__("torch").tensor(fa)).float().mean())

state = load_json("e1_layer_grid", name) or {}
grid = state.get("grid", {})
timing = state.get("timing", {})
for dname, D in dirs.items():
    if dname in grid:
        print(f"[e1] {dname}: already done, skipping", flush=True)
        continue
    configs = [Config(f"{dname} L{L} x{s:g}", vec=D[L], strength=s, layers=(L,), meta={"layer": L, "strength": s})
               for L in layers for s in strengths]
    with Timer(lm.device) as tg:
        replies = sweep.generate(configs, prompts,
                                 on_chunk=lambda n: print(f"[e1] {dname} chunk {n}", flush=True) if n % 5 == 0 else None)
    with Timer(lm.device) as td:
        acc, kl = sweep.fact_damage(configs, fp, fa, before)
    with Timer() as ts:
        moods = [avg(mood(r)) for r in replies]
    grid[dname] = [{"layer": c.meta["layer"], "strength": c.meta["strength"], "mood": r3(m), "facts": r3(a), "kl": r3(k)}
                   for c, m, a, k in zip(configs, moods, acc, kl)]
    timing[dname] = {"configs": len(configs), "rows": len(configs) * len(prompts), "generate_s": r3(tg.seconds),
                     "damage_s": r3(td.seconds), "score_s": r3(ts.seconds),
                     "configs_per_s": r3(len(configs) / max(1e-9, tg.seconds))}
    print(f"[e1] {dname}: {timing[dname]}", flush=True)
    best = sorted(range(len(configs)), key=lambda i: moods[i])[:5]
    append_samples("e1_layer_grid", [{"run": name, "config": configs[i].name, "mood": r3(moods[i]), "prompt": pr, "reply": r}
                                     for i in best for pr, r in list(zip(prompts, replies[i]))[:3]], name=name + "_samples")
    save_json("e1_layer_grid", name, {"args": vars(args), "layers": layers, "strengths": strengths, "topics": topics,
                                      "facts_baseline": r3(base_acc), "grid": grid, "timing": timing})
sweep.close()

sad_rows = grid["sad"]
win = min(sad_rows, key=lambda r: r["mood"] + (1.0 if r["kl"] > 0.5 else 0))
print("[e1] saddest setting with KL<=0.5:", win)
print("[e1] total configs:", sum(t["configs"] for t in timing.values()))
