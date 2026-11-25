"""E4: the refusal direction. One direction (harmful mean - harmless mean at the last prompt token) that switches
refusal on and off (Arditi et al. 2024, "Refusal in language models is mediated by a single direction").

All batched, chosen on the train halves of the prompt lists, then checked on the test halves:
  1. baseline refusal rate on harmful and harmless prompts (no edit)
  2. induce: push +s * d[L] on harmless prompts, every layer x --strengths -> does the model refuse cake recipes?
  3. remove: project out the layer-L direction after every block (--ablates) on harmful prompts -> refusal rate + KL
  4. controls: a random direction with the same length (both sweeps), and a shuffled-label direction at the winners
  5. the induce and remove winners (and their controls) on the test prompts

Safety: replies to harmful prompts under the remove edit are stored as the first 60 characters only, plus the
refused flag. Full text is kept for harmless prompts.
"""
import random

import torch

from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering.directions import random_like, refusal_acts
from batched_steering.io import append_samples, load_json, r3, save_json
from batched_steering.score import avg, facts, refused
from batched_steering.data.refusal_prompts import HARMFUL_TEST, HARMFUL_TRAIN, HARMLESS_TEST, HARMLESS_TRAIN

p = parser(__doc__)
p.add_argument("--strengths", default="0.5,1,1.5,2,3")
p.add_argument("--ablates", default="0.5,1")
p.add_argument("--max-kl", type=float, default=0.2, help="winners must keep fact KL below this")
p.set_defaults(tokens=40)
args = p.parse_args()
lm = setup(args)
N = lm.n_layers
n = args.topics
harm_tr, harm_te, ok_tr, ok_te = (limit(x, n) for x in (HARMFUL_TRAIN, HARMFUL_TEST, HARMLESS_TRAIN, HARMLESS_TEST))
name = run_name(args, f"e4_{lm.device.type}_{args.dtype}_{args.eager}")

with Timer() as t_dir:
    bad, good = refusal_acts(lm, harm_tr, ok_tr)
    D = bad.mean(0) - good.mean(0)                                     # (L+1, d)
    R = random_like(D, seed=1)
    acts = torch.cat([bad, good])
    perm = torch.randperm(len(acts), generator=torch.Generator().manual_seed(0))
    S = acts[perm[:len(bad)]].mean(0) - acts[perm[len(bad):]].mean(0)  # shuffled labels
    S = S / S.norm(dim=-1, keepdim=True) * D.norm(dim=-1, keepdim=True)
print(f"[e4] directions in {t_dir.seconds:.1f}s; norms L8/L14/L20: "
      + " / ".join(f"{D[L].norm():.1f}" for L in (8, 14, 20) if L <= N), flush=True)
dirs = {"refusal": D, "random": R, "shuffled": S}


def push(dname, L, s):
    return Config(f"induce {dname} L{L} x{s:g}", vec=dirs[dname][L], strength=s, layers=(L,),
                  meta={"direction": dname, "layer": L, "strength": s})


def remove(dname, L, a):
    return Config(f"remove {dname} L{L} x{a:g}", unit=dirs[dname][L], ablate=a, layers=tuple(range(1, N + 1)),
                  meta={"direction": dname, "layer": L, "ablate": a})


fp, fa = facts(lm.tok)
sweep = Sweep(lm, args.batch, args.tokens, args.temperature)
before = sweep.last_logprobs([Config()], fp)[0]


def run(configs, prompts, tag):
    with Timer(lm.device) as tg:
        replies = sweep.generate(configs, prompts,
                                 on_chunk=lambda k: print(f"[e4] {tag} chunk {k}", flush=True) if k % 10 == 0 else None)
    with Timer(lm.device) as td:
        acc, kl = sweep.fact_damage(configs, fp, fa, before)
    rows = [{**c.meta, "refusal": r3(avg(refused(r) for r in rs)), "facts": r3(float(a)), "kl": r3(float(k))}
            for c, rs, a, k in zip(configs, replies, acc, kl)]
    t = {"configs": len(configs), "rows": len(configs) * len(prompts), "generate_s": r3(tg.seconds),
         "damage_s": r3(td.seconds), "configs_per_s": r3(len(configs) / max(1e-9, tg.seconds))}
    print(f"[e4] {tag}: {t}", flush=True)
    return rows, replies, t


state = load_json("e4_refusal", name) or {}
timing = state.get("timing", {})
strengths = [float(x) for x in args.strengths.split(",")]
ablates = [float(x) for x in args.ablates.split(",")]


def checkpoint():
    save_json("e4_refusal", name, {"args": vars(args), **state, "timing": timing})


# 1. baseline
if "baseline" not in state:
    out = {}
    for tag, ps in (("harmful_train", harm_tr), ("harmless_train", ok_tr), ("harmful_test", harm_te), ("harmless_test", ok_te)):
        rs = sweep.generate([Config("none")], ps)[0]
        out[tag] = r3(avg(refused(r) for r in rs))
        append_samples("e4_refusal", [{"run": name, "config": "none", "prompts": tag, "prompt": pr,
                                       "reply": r[:60] if tag.startswith("harmful") else r, "refused": refused(r)}
                                      for pr, r in list(zip(ps, rs))[:6]], name=name + "_samples")
    state["baseline"] = out
    print(f"[e4] baseline refusal: {out}", flush=True)
    checkpoint()

# 2. induce on harmless prompts
if "induce" not in state:
    configs = [push(dn, L, s) for dn in ("refusal", "random") for L in range(1, N + 1) for s in strengths]
    rows, _, timing["induce"] = run(configs, ok_tr, "induce")
    state["induce"] = rows
    checkpoint()

# 3. remove on harmful prompts
if "remove" not in state:
    configs = [remove(dn, L, a) for dn in ("refusal", "random") for L in range(1, N + 1) for a in ablates]
    rows, _, timing["remove"] = run(configs, harm_tr, "remove")
    state["remove"] = rows
    checkpoint()

# 5. winners on the test prompts (+ the same settings with random and shuffled directions)
ok_kl = lambda rows: [r for r in rows if r["direction"] == "refusal" and r["kl"] < args.max_kl] or \
    [r for r in rows if r["direction"] == "refusal"]
wi = max(ok_kl(state["induce"]), key=lambda r: (r["refusal"], -r["kl"]))
wr = min(ok_kl(state["remove"]), key=lambda r: (r["refusal"], r["kl"]))
induce_cfgs = [push(dn, wi["layer"], wi["strength"]) for dn in dirs]
remove_cfgs = [remove(dn, wr["layer"], wr["ablate"]) for dn in dirs]
ri, rep_i, _ = run(induce_cfgs, ok_te, "test induce")
rr, rep_r, _ = run(remove_cfgs, harm_te, "test remove")
state["test"] = {"induce": {"train": wi, "test": {dn: r for dn, r in zip(dirs, ri)}},
                 "remove": {"train": wr, "test": {dn: r for dn, r in zip(dirs, rr)}}}
append_samples("e4_refusal", [{"run": name, "config": induce_cfgs[0].name, "prompts": "harmless_test", "prompt": pr,
                               "reply": r, "refused": refused(r)} for pr, r in list(zip(ok_te, rep_i[0]))[:8]]
               + [{"run": name, "config": remove_cfgs[0].name, "prompts": "harmful_test", "prompt": pr,
                   "reply": r[:60], "refused": refused(r)} for pr, r in list(zip(harm_te, rep_r[0]))[:8]],
               name=name + "_samples")
checkpoint()
sweep.close()

b, T = state["baseline"], state["test"]
print(f"[e4] baseline: harmful {b['harmful_test']:.2f} / harmless {b['harmless_test']:.2f} refused (test)")
print(f"[e4] induce winner L{wi['layer']} x{wi['strength']:g}: train {wi['refusal']:.2f} -> test "
      + ", ".join(f"{dn} {r['refusal']:.2f}" for dn, r in T['induce']['test'].items()))
print(f"[e4] remove winner L{wr['layer']} x{wr['ablate']:g}: train {wr['refusal']:.2f} (KL {wr['kl']:.3f}) -> test "
      + ", ".join(f"{dn} {r['refusal']:.2f} (KL {r['kl']:.3f})" for dn, r in T['remove']['test'].items()))
print("[e4] done")
