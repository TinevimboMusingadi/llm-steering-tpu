"""M0: hello TPU. Load Qwen3, generate, then check hooks: read-only, zero push, real push at layer 12.
Done when: zero push == no hook (identical tokens) and a non-zero push changes the text. Also times Strict vs DeferAndFuse."""
from _common import parser, run_name, setup

import torch

from batched_steering import Generator, Timer, chat, decode, encode, set_eager_mode
from batched_steering.directions import sad_direction
from batched_steering.io import r3, save_json
from batched_steering.steer import SingleSteer

args = parser(__doc__).parse_args()
lm = setup(args)
prompts = ["Write two sentences about a dog.", "Write two sentences about the sea.",
           "Write two sentences about breakfast.", "Write two sentences about a train station."]
ids, mask, pos = encode(lm.tok, [chat(lm.tok, p) for p in prompts], lm.device)
gen = Generator(lm, len(prompts), ids.shape[1], args.tokens)
res = {}

with Timer(lm.device) as t:
    base = gen(ids, mask, pos, temperature=0)
res["first_generate_s"] = r3(t.seconds)
with Timer(lm.device) as t:
    base2 = gen(ids, mask, pos, temperature=0)
res["second_generate_s"] = r3(t.seconds)
texts = decode(lm.tok, base)
print("[m0] baseline:", texts[0][:120].replace("\n", " "))
res["deterministic"] = bool(torch.equal(base.cpu(), base2.cpu()))

# read-only hook
seen = {}
def read_only(m, i, o):  # must return None: any return value replaces the block output
    seen.setdefault("shape", tuple((o[0] if isinstance(o, tuple) else o).shape))


h = lm.layers[11].register_forward_hook(read_only)
gen(ids, mask, pos, temperature=0)
h.remove()
res["read_hook_shape"] = seen.get("shape")

# zero push
with SingleSteer(lm.layers) as s:
    s.add(12, torch.zeros(lm.d_model))
    zero = gen(ids, mask, pos, temperature=0)
res["zero_push_identical"] = bool(torch.equal(zero.cpu(), base.cpu()))

# real push: 2x the sad direction at layer 12
d = sad_direction(lm)
with SingleSteer(lm.layers) as s:
    s.add(12, 2 * d[12])
    pushed = gen(ids, mask, pos, temperature=0)
ptexts = decode(lm.tok, pushed)
res["push_changes_text"] = ptexts != texts
res["sad_norm_L12"] = r3(d[12].norm())
res["samples"] = {"baseline": texts, "sad_2x_L12": ptexts}
print("[m0] sad 2x  :", ptexts[0][:120].replace("\n", " "))

# eager modes (TPU only)
if lm.device.type == "tpu":
    for mode in ("strict", "defer"):
        set_eager_mode(mode)
        gen(ids, mask, pos, temperature=0)
        with Timer(lm.device) as t:
            gen(ids, mask, pos, temperature=0)
        res[f"generate_s_{mode}"] = r3(t.seconds)
    set_eager_mode(args.eager)

print("[m0]", {k: v for k, v in res.items() if k != "samples"})
assert res["zero_push_identical"], "zero push changed the output"
assert res["push_changes_text"], "push did not change the output"
save_json("m0_hello", run_name(args, f"m0_{lm.device.type}"), {"args": vars(args), **res})
print("[m0] PASS")
