"""E3: removal random search, 1,000 trials instead of 40, and the winner's curse.

Reference (avbiswas 03): each trial picks which layer's sad direction to use (4..24), how much of it to remove
(0.2..1.2) and a range of blocks (first 1..20, last first+1..28). Scored on the tune topics with "Write two sad
sentences about <topic>": mood (higher = the removal worked) and KL on 40 facts (lower = less damage). Winner = the
least damaging trial with mood > 0.8, then checked on the test topics. Control: the same search with random
directions.

Here: the same search, batched, with --trials trials (default 1000; the first 40 use the reference's RNG stream, so
"40 trials" is a prefix of ours). Then on the test topics:
  - the 40-trial winner and the 1000-trial winner
  - the hand-picked edit (layer-12 direction, fully removed after every block)
  - the top --recheck trials by tune score, to see how much of the tune score survives on new topics (winner's curse)
"""
import random

from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import random_like, sad_acts
from batched_steering.io import append_samples, load_json, r3, save_json
from batched_steering.score import avg, facts, mood
from batched_steering.data.topics import TEST, TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--trials", type=int, default=1000)
p.add_argument("--recheck", type=int, default=40, help="top trials re-run on the test topics")
p.add_argument("--threshold", type=float, default=0.8, help="winner = lowest KL with tune mood above this")
p.add_argument("--hand-layer", type=int, default=12)
args = p.parse_args()
lm = setup(args)
N = lm.n_layers

tune = make_prompts(limit(TUNE, args.topics), "sad")
test = make_prompts(limit(TEST, args.topics), "sad")
name = run_name(args, f"e3_{lm.device.type}_{args.dtype}_{args.eager}")

happy, sad = sad_acts(lm)
all_dirs = sad.mean(0) - happy.mean(0)                                 # (L+1, d)

rng = random.Random(args.seed)
picks = []
for _ in range(args.trials):                                           # same draw order as the reference
    src = rng.randrange(4, 25)
    strength = rng.uniform(0.2, 1.2)
    first = rng.randrange(1, 21); last = rng.randrange(first + 1, 29)
    picks.append((src, strength, first, min(last, N)))


def cfg(direction, src, strength, first, last, tag):
    return Config(f"{tag} dir L{src} x{strength:.2f} blocks {first}-{last}", unit=direction, ablate=strength,
                  layers=tuple(range(first, last + 1)),
                  meta={"direction_layer": src, "strength": r3(strength), "blocks": [first, last]})


fp, fa = facts(lm.tok)
sweep = Sweep(lm, args.batch, args.tokens, args.temperature)
before = sweep.last_logprobs([Config()], fp)[0]


def evaluate(configs, ps, tag):
    with Timer(lm.device) as tg:
        replies = sweep.generate(configs, ps, on_chunk=lambda n: print(f"[e3] {tag} chunk {n}", flush=True) if n % 10 == 0 else None)
    with Timer(lm.device) as td:
        acc, kl = sweep.fact_damage(configs, fp, fa, before)
    with Timer() as ts:
        moods = [avg(mood(r)) for r in replies]
    t = {"configs": len(configs), "rows": len(configs) * len(ps), "generate_s": r3(tg.seconds), "damage_s": r3(td.seconds),
         "score_s": r3(ts.seconds), "configs_per_s": r3(len(configs) / max(1e-9, tg.seconds))}
    return [{**c.meta, "mood": r3(m), "facts": r3(float(a)), "kl": r3(float(k))} for c, m, a, k in zip(configs, moods, acc, kl)], replies, t


state = load_json("e3_removal_search", name) or {}
timing = state.get("timing", {})
searches = {"sad": lambda L: all_dirs[L], "random": lambda L: random_like(all_dirs[L], seed=L)}
for sname, pick_dir in searches.items():
    if sname in state.get("search", {}):
        print(f"[e3] {sname} search: already done", flush=True)
        continue
    configs = [cfg(pick_dir(src), src, s, f, l, sname) for src, s, f, l in picks]
    rows, _, t = evaluate(configs, tune, sname)
    state.setdefault("search", {})[sname] = rows
    timing[sname] = t
    print(f"[e3] {sname} search: {t}", flush=True)
    save_json("e3_removal_search", name, {"args": vars(args), **state, "timing": timing})


def winner(rows):
    good = [(i, r) for i, r in enumerate(rows) if r["mood"] > args.threshold]
    if not good:   # fall back to the happiest trial
        return max(enumerate(rows), key=lambda x: x[1]["mood"])
    return min(good, key=lambda x: x[1]["kl"])


real = state["search"]["sad"]
w40, w_all = winner(real[:40]), winner(real)
# winner's curse: the best trials by tune mood among the low-damage ones, ranked like the winner rule
ranked = sorted(range(len(real)), key=lambda i: (real[i]["mood"] <= args.threshold, real[i]["kl"] if real[i]["mood"] > args.threshold else -real[i]["mood"]))
recheck = list(dict.fromkeys([w40[0], w_all[0]] + ranked[:args.recheck]))
configs = [cfg(all_dirs[picks[i][0]], *picks[i], "recheck") for i in recheck]
configs.append(Config("hand-picked", unit=all_dirs[args.hand_layer], ablate=1.0, layers=tuple(range(1, N + 1)),
                      meta={"direction_layer": args.hand_layer, "strength": 1.0, "blocks": [1, N]}))
rows, replies, t = evaluate(configs, test, "test")
timing["test"] = t
state["test"] = {"rows": [{"trial": (recheck + [None])[j], "tune": real[recheck[j]] if j < len(recheck) else None, "test": r}
                          for j, r in enumerate(rows)],
                 "winner_40": {"trial": w40[0], "tune": w40[1], "test": rows[recheck.index(w40[0])]},
                 "winner_all": {"trial": w_all[0], "tune": w_all[1], "test": rows[recheck.index(w_all[0])]},
                 "hand_picked": rows[-1]}
append_samples("e3_removal_search", [{"run": name, "config": configs[j].name, "prompt": pr, "reply": r}
                                     for j in (recheck.index(w40[0]), recheck.index(w_all[0]), len(configs) - 1)
                                     for pr, r in list(zip(test, replies[j]))[:4]], name=name + "_samples")
save_json("e3_removal_search", name, {"args": vars(args), **state, "timing": timing, "reference": {
    "trials": 40, "winner": {"direction_layer": 13, "strength": 1.168, "blocks": [12, 19], "tune_mood": 0.868, "kl": 0.042},
    "test": {"optimized": 0.822, "hand_picked": 0.353}}})
sweep.close()

T = state["test"]
for k in ("winner_40", "winner_all"):
    print(f"[e3] {k}: {T[k]['tune']} -> test mood {T[k]['test']['mood']:+.2f}")
print(f"[e3] hand-picked on test: mood {T['hand_picked']['mood']:+.2f} KL {T['hand_picked']['kl']:.3f}")
gap = avg(r["tune"]["mood"] - r["test"]["mood"] for r in T["rows"] if r["tune"])
print(f"[e3] mean tune->test mood drop over {len(recheck)} rechecked trials: {gap:+.3f}")
print(f"[e3] random-direction search: best tune mood {max(r['mood'] for r in state['search']['random']):+.2f}")
print("[e3] done")
