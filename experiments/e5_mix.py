"""E5: mixing two directions. Push sad (every token, E1 winner layers) and tomato (generated tokens only, E2 winner
settings) at the same time, in the same row, and score each reply for BOTH mood and tomato success.

Question: do two pushes simply add up, or do they get in each other's way?
  - grid: sad strength x tomato strength, for two layer pairings, on the choose topics and the test topics
  - interaction = mixed effect - (sad-alone effect + tomato-alone effect), for mood and for tomato success
  - cosine between the two directions at every layer
  - sanity: sad push + removal of the sad direction after the same blocks should cancel the push

Uses the second push slot of BatchedSteer (Config.extra); tested against one-at-a-time hooks in tests/.
"""
import torch

from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import sad_direction, tomato_direction
from batched_steering.io import append_samples, load_json, r3, save_json
from batched_steering.score import avg, mood, tomato_success
from batched_steering.data.topics import TEST, TRAIN, TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--sad", default="0,0.5,1,1.5,2")
p.add_argument("--tomato", default="0,5,10,12.3,15")
p.set_defaults(tokens=50)
args = p.parse_args()
lm = setup(args)
N = lm.n_layers
name = run_name(args, f"e5_{lm.device.type}_{args.dtype}_{args.eager}")

sad, tom = sad_direction(lm), tomato_direction(lm)
cos = torch.nn.functional.cosine_similarity(sad, tom, dim=-1)
print("[e5] cos(sad, tomato) L6/L12/L17/L18/L24: " + " / ".join(f"{cos[L]:+.3f}" for L in (6, 12, 17, 18, 24) if L <= N), flush=True)

# (name, sad layer, tomato layer, tomato spread): E1 winners L12 / L17, E2 winners L18 x10 / L17 x12.3 spread 2
PAIRS = [("sad L12 + tomato L18", min(12, N), min(18, N), 1), ("sad L17 + tomato L17 spread 2", min(17, N), min(17, N), 2)]
S = [float(x) for x in args.sad.split(",")]
Tm = [float(x) for x in args.tomato.split(",")]


def cfg(pair, s, t):
    pname, Ls, Lt, k = pair
    tomato = Config(vec=tom[Lt] / k, strength=t, layers=tuple(range(Lt, min(Lt + k, N + 1))), gen_only=True)
    return Config(f"{pname}: sad x{s:g} tomato x{t:g}", vec=sad[Ls], strength=s, layers=(Ls,), extra=[tomato],
                  meta={"pair": pname, "sad": s, "tomato": t})


sanity = [Config("sad L12 x2", vec=sad[12], strength=2.0, layers=(12,), meta={"sanity": "push"}),
          Config("remove sad after 12..N", unit=sad[12], ablate=1.0, layers=tuple(range(12, N + 1)), meta={"sanity": "remove"}),
          Config("sad L12 x2 + remove", vec=sad[12], strength=2.0, unit=sad[12], ablate=1.0,
                 layers=(12,), extra=[], meta={"sanity": "push+remove at L12"})]
# the push+remove row: Config layers drive both push and removal; push only at 12, removal only at 12 -> exactly cancels
# the push and also removes the prompt's own sad component at L12

sweep = Sweep(lm, args.batch, args.tokens, args.temperature)
state = load_json("e5_mix", name) or {}
timing = state.get("timing", {})
configs = [cfg(pr, s, t) for pr in PAIRS for s in S for t in Tm]

for split, topics in (("choose", limit(TRAIN + TUNE, args.topics)), ("test", limit(TEST, args.topics))):
    if split in state:
        continue
    with Timer(lm.device) as tg:
        replies = sweep.generate(configs + sanity, make_prompts(topics),
                                 on_chunk=lambda k: print(f"[e5] {split} chunk {k}", flush=True) if k % 10 == 0 else None)
    with Timer() as ts:
        moods = [avg(mood(rs)) for rs in replies]
        toms = [avg(tomato_success(r, t) for r, t in zip(rs, topics)) for rs in replies]
    rows = [{**c.meta, "mood": r3(m), "tomato_success": r3(x)} for c, m, x in zip(configs + sanity, moods, toms)]
    grid, sane = rows[:len(configs)], rows[len(configs):]
    # interaction: mixed - (sad alone + tomato alone - none), per pair
    for g in grid:
        same = [r for r in grid if r["pair"] == g["pair"]]
        at = lambda s, t: next(r for r in same if r["sad"] == s and r["tomato"] == t)
        base, a, b = at(S[0], Tm[0]), at(g["sad"], Tm[0]), at(S[0], g["tomato"])
        for k in ("mood", "tomato_success"):
            g[f"{k}_interaction"] = r3(g[k] - (a[k] + b[k] - base[k]))
    state[split] = {"grid": grid, "sanity": sane}
    timing[split] = {"configs": len(configs) + len(sanity), "rows": (len(configs) + len(sanity)) * len(topics),
                     "generate_s": r3(tg.seconds), "score_s": r3(ts.seconds),
                     "configs_per_s": r3((len(configs) + len(sanity)) / max(1e-9, tg.seconds))}
    print(f"[e5] {split}: {timing[split]}", flush=True)
    pick = [i for i, c in enumerate(configs) if c.meta["sad"] in (S[0], S[-1]) and c.meta["tomato"] in (Tm[0], Tm[2])]
    append_samples("e5_mix", [{"run": name, "split": split, "config": configs[i].name, "topic": t, "reply": r}
                              for i in pick for t, r in list(zip(topics, replies[i]))[:3]], name=name + "_samples")
    save_json("e5_mix", name, {"args": vars(args), "cos_sad_tomato": [r3(float(x)) for x in cos], **state, "timing": timing})
sweep.close()

for split in ("choose", "test"):
    for pr in PAIRS:
        g = [r for r in state[split]["grid"] if r["pair"] == pr[0]]
        at = lambda s, t: next(r for r in g if r["sad"] == s and r["tomato"] == t)
        n, a, b, m = at(S[0], Tm[0]), at(S[-1], Tm[0]), at(S[0], Tm[2]), at(S[-1], Tm[2])
        print(f"[e5] {split} {pr[0]}: none mood {n['mood']:+.2f} tom {n['tomato_success']:.2f} | sad x{S[-1]:g} "
              f"{a['mood']:+.2f}/{a['tomato_success']:.2f} | tomato x{Tm[2]:g} {b['mood']:+.2f}/{b['tomato_success']:.2f} | "
              f"both {m['mood']:+.2f}/{m['tomato_success']:.2f}")
    print(f"[e5] {split} sanity: " + ", ".join(f"{r['sanity']} {r['mood']:+.2f}" for r in state[split]["sanity"]))
print("[e5] done")
