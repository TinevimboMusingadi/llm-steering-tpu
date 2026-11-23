"""E2: the tomato map. Where (layer) and how hard (strength) must you push to get tomatoes into every answer
without losing the topic?

Reference (avbiswas 03b): by-hand grid of 5 layers x 4 strengths + 60 random settings, chosen on 36 topics
(train + tune), compared on the 24 test topics. Push only while the model writes (decode steps), optionally spread
over 1-5 blocks (strength / spread on each).

Here, batched:
  1. full map: every layer x strengths 2..20 (spread 1) on the 36 choosing topics
  2. random search: --trials settings (layer, strength, spread), same ranges as the reference
  3. on the test topics: the reference's by-hand pick, our map's best, the 60-trial winner and the full-search winner
"""
import random

from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import tomato_direction
from batched_steering.io import append_samples, load_json, r3, save_json
from batched_steering.score import avg, tomato_success
from batched_steering.data.topics import TEST, TRAIN, TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--strengths", default="2,4,5,6,8,10,12,14,15,16,18,20")
p.add_argument("--trials", type=int, default=1000)
p.set_defaults(tokens=50)
args = p.parse_args()
lm = setup(args)
N = lm.n_layers

choose = limit(TRAIN + TUNE, args.topics)
test = limit(TEST, args.topics)
strengths = [float(s) for s in args.strengths.split(",")]
name = run_name(args, f"e2_{lm.device.type}_{args.dtype}_{args.eager}")

with Timer() as t_dir:
    d = tomato_direction(lm)                                          # (L+1, d)
print(f"[e2] tomato direction in {t_dir.seconds:.1f}s, norms L6/L18: {d[6].norm():.1f} / {d[18].norm():.1f}", flush=True)


def cfg(L, s, k=1):
    blocks = tuple(range(L, min(L + k, N + 1)))
    return Config(f"tomato L{L} x{s:.2f} spread{k}", vec=d[L] / k, strength=s, layers=blocks, gen_only=True,
                  meta={"layer": L, "strength": r3(s), "spread": k})


sweep = Sweep(lm, args.batch, args.tokens, args.temperature)


def score(configs, topics, tag):
    with Timer(lm.device) as tg:
        replies = sweep.generate(configs, make_prompts(topics),
                                 on_chunk=lambda n: print(f"[e2] {tag} chunk {n}", flush=True) if n % 10 == 0 else None)
    succ = [avg(tomato_success(r, t) for r, t in zip(rs, topics)) for rs in replies]
    return succ, replies, tg.seconds


state = load_json("e2_tomato_map", name) or {}
timing = state.get("timing", {})

# 1. full map
if "map" not in state:
    configs = [cfg(L, s) for L in range(1, N + 1) for s in strengths]
    succ, replies, secs = score(configs, choose, "map")
    state["map"] = [{**c.meta, "success": r3(x)} for c, x in zip(configs, succ)]
    timing["map"] = {"configs": len(configs), "rows": len(configs) * len(choose), "generate_s": r3(secs),
                     "configs_per_s": r3(len(configs) / max(1e-9, secs))}
    best = sorted(range(len(configs)), key=lambda i: -succ[i])[:5]
    append_samples("e2_tomato_map", [{"run": name, "config": configs[i].name, "success": r3(succ[i]), "topic": t, "reply": r}
                                     for i in best for t, r in list(zip(choose, replies[i]))[:4]], name=name + "_samples")
    print(f"[e2] map: {timing['map']}", flush=True)
    save_json("e2_tomato_map", name, {"args": vars(args), **state, "timing": timing})

# 2. random search (same ranges as the reference)
if "trials" not in state:
    rng = random.Random(args.seed)
    picks = [(rng.randrange(round(N * 0.15), round(N * 0.9)), rng.uniform(4, 20), rng.randrange(1, 6)) for _ in range(args.trials)]
    configs = [cfg(*x) for x in picks]
    succ, _, secs = score(configs, choose, "search")
    state["trials"] = [{**c.meta, "success": r3(x)} for c, x in zip(configs, succ)]
    timing["search"] = {"configs": len(configs), "rows": len(configs) * len(choose), "generate_s": r3(secs),
                        "configs_per_s": r3(len(configs) / max(1e-9, secs))}
    print(f"[e2] search: {timing['search']}", flush=True)
    save_json("e2_tomato_map", name, {"args": vars(args), **state, "timing": timing})

# 3. winners on the test topics
hand_grid = [r for r in state["map"] if r["layer"] in [round(N * f) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
             and r["strength"] in (5, 10, 15, 20) and r["spread"] == 1]
pick = lambda rows: max(rows, key=lambda r: r["success"])
winners = {"by hand (5 layers x 4 round strengths)": pick(hand_grid) if hand_grid else pick(state["map"]),
           "full map best": pick(state["map"]),
           "search best, first 60 trials": pick(state["trials"][:60]),
           f"search best, {len(state['trials'])} trials": pick(state["trials"])}
configs = [cfg(w["layer"], w["strength"], w["spread"]) for w in winners.values()]
succ, replies, _ = score(configs, test, "test")
state["test"] = {k: {**w, "choose_success": w["success"], "test_success": r3(s)}
                 for (k, w), s in zip(winners.items(), succ)}
state["test_examples"] = {k: dict(list(zip(test, rs))[:6]) for k, rs in zip(winners, replies)}
for k, v in state["test"].items():
    print(f"[e2] {k:40s} L{v['layer']} x{v['strength']} spread{v['spread']}: choose {v['choose_success']:.2f} -> test {v['test_success']:.2f}")
save_json("e2_tomato_map", name, {"args": vars(args), **state, "timing": timing, "reference": {
    "L18": {5: 0.0, 10: 0.44, 15: 0.25, 20: 0.03}, "by_hand_test": 0.42, "search_test": 0.21, "search_choose": 0.47}})
sweep.close()
print("[e2] done")
