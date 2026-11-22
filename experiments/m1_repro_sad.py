"""M1/M2: reproduce the reference sad-direction result, one config at a time (--mode loop) or batched (--mode batched).

Sad direction at layer 12, strengths -3..+4 on the 24 test topics, plus two same-length controls (random, shuffled labels)
at 2x; fact damage at 2x; layer sweep at 2x on the 12 tune topics. Compared against batched_steering/reference.py.
"""
from _common import limit, parser, run_name, setup

from batched_steering import Config, Sweep, Timer
from batched_steering import reference as ref
from batched_steering.directions import match_norm, random_like, sad_acts, shuffled_direction
from batched_steering.io import append_samples, r3, save_json
from batched_steering.loop import Loop
from batched_steering.score import avg, facts, mood
from batched_steering.data.topics import TEST, TUNE, prompts as make_prompts

p = parser(__doc__)
p.add_argument("--mode", default="batched", choices=["loop", "batched"])
p.add_argument("--layer", type=int, default=12)
p.add_argument("--no-layer-sweep", action="store_true")
args = p.parse_args()
lm = setup(args)
L = args.layer

with Timer(lm.device) as t_dir:
    happy, sad = sad_acts(lm)
    d = sad.mean(0) - happy.mean(0)                           # (L+1, d)
push2 = 2 * d[L]
controls = {"random direction": random_like(push2), "shuffled labels": match_norm(shuffled_direction(happy, sad)[L], push2)}

strengths = [-3, -2, -1, 0, 1, 2, 3, 4]
configs = [Config(f"sad x{s:+d}", vec=d[L], strength=float(s), layers=(L,), meta={"strength": s}) for s in strengths]
configs += [Config(name, vec=v, strength=1.0, layers=(L,), meta={"control": name}) for name, v in controls.items()]
layer_cfgs = [] if args.no_layer_sweep else [Config(f"L{l} x2", vec=d[l], strength=2.0, layers=(l,), meta={"layer": l})
                                              for l in range(2, 27, 2)]

test, tune = limit(make_prompts(TEST), args.topics), limit(make_prompts(TUNE), args.topics)
runner = Loop(lm, args.tokens, args.temperature) if args.mode == "loop" else Sweep(lm, args.batch, args.tokens, args.temperature)

with Timer(lm.device) as t_gen:
    replies = runner.generate(configs, test)
with Timer(lm.device) as t_layer:
    layer_replies = runner.generate(layer_cfgs, tune) if layer_cfgs else []

fp, fa = facts(lm.tok)
base_runner = Sweep(lm, len(fp), args.tokens) if args.mode == "loop" else runner
before = base_runner.last_logprobs([Config()], fp)[0]
if base_runner is not runner:
    base_runner.close()
acc, kl = runner.fact_damage([Config(), configs[strengths.index(2)]], fp, fa, before)
if hasattr(runner, "close"):
    runner.close()

with Timer() as t_score:
    moods = [avg(mood(r)) for r in replies]
    layer_moods = [avg(mood(r)) for r in layer_replies]

curve = {s: r3(m) for s, m in zip(strengths, moods)}
ctrl = {c.meta["control"]: r3(m) for c, m in zip(configs[len(strengths):], moods[len(strengths):])}
sweep = {c.meta["layer"]: r3(m) for c, m in zip(layer_cfgs, layer_moods)}
damage = {"facts_before": r3(acc[0]), "facts_after": r3(acc[1]), "kl": r3(kl[1])}
n_rows = len(configs) * len(test) + len(layer_cfgs) * len(tune)
timing = {"directions_s": r3(t_dir.seconds), "generate_s": r3(t_gen.seconds), "layer_sweep_s": r3(t_layer.seconds),
          "score_s": r3(t_score.seconds), "configs": len(configs) + len(layer_cfgs), "rows": n_rows,
          "configs_per_s": r3((len(configs) + len(layer_cfgs)) / max(1e-9, t_gen.seconds + t_layer.seconds))}

print("[m1] strength curve:", curve, "\n     reference     :", ref.E01_STRENGTH_CURVE_L12)
print("[m1] controls:", ctrl, "| reference", ref.E01_CONTROLS_2X)
print("[m1] damage at 2x:", damage, "| reference", ref.E01_DAMAGE_2X)
print("[m1] layer sweep:", sweep, "\n     reference  :", ref.E01_LAYER_SWEEP_2X_TUNE)
print("[m1] timing:", timing)

name = run_name(args, f"m1_{args.mode}_{lm.device.type}_{args.eager}")
save_json("m1_repro_sad", name, {"args": vars(args), "layer": L, "strength_curve": curve, "controls": ctrl,
                                 "damage_at_2x": damage, "layer_sweep_2x_tune": sweep, "timing": timing,
                                 "reference": {"strength_curve": ref.E01_STRENGTH_CURVE_L12, "controls": ref.E01_CONTROLS_2X,
                                               "damage": ref.E01_DAMAGE_2X, "layer_sweep": ref.E01_LAYER_SWEEP_2X_TUNE}})
append_samples("m1_repro_sad", [{"run": name, "config": c.name, "prompt": pr, "reply": r}
                                for c, rs in zip(configs, replies) for pr, r in list(zip(test, rs))[:3]], name=name + "_samples")
ok = curve[0] > 0.2 and curve[2] < curve[0] - 0.3 and ctrl["random direction"] > curve[2] + 0.2
print("[m1] trend reproduced:", ok)
