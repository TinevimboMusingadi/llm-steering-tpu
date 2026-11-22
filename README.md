# llm-steering-tpu (lab)

Private lab repo for **Batched LLM Steering on TPU** (TorchTPU Sprint 26H4). Every batch row runs its own steering experiment (direction, layer, strength, removal), so one forward pass covers hundreds of settings. Answer key: [avbiswas/learn-mech-interp `03-steering`](https://github.com/avbiswas/learn-mech-interp/tree/main/03-steering) (MIT); the data in `batched_steering/data/` is ported from it.

```
batched_steering/   device, model, directions, steer (BatchedSteer/SingleSteer), sweep, generate, loop, score, io, reference
experiments/        m0_hello, m1_repro_sad (--mode loop|batched), m2_batched_vs_loop  (all take --tiny for CPU smoke runs)
tests/              CPU equivalence tests (tiny random Qwen3)
docs/LOG.md         what broke and how it was fixed
results/            JSON + samples from TPU runs
```

Local check (CPU):
```
HF_HUB_OFFLINE=1 python -m pytest tests -q
python experiments/m2_batched_vs_loop.py --tiny --device cpu --tokens 8 --topics 3 --batch 16
```
On TPU: `pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu`, then run the experiment scripts (the device is picked automatically; `--eager defer` selects DeferAndFuse).
