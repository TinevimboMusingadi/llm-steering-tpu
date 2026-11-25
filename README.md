# llm-steering-tpu (lab)

Lab repo for **Batched LLM Steering on TPU** (TorchTPU Sprint 26H4). Every batch row runs its own steering experiment (direction, layer, strength, removal, and an optional second push), so one forward pass covers hundreds of settings. All experiments ran on a single TPU v5e chip with Qwen3-0.6B.

- **Write-up (blog):** https://tinevimbomusingadi.github.io/steering-blog/
- **Answer key:** [avbiswas/learn-mech-interp `03-steering`](https://github.com/avbiswas/learn-mech-interp/tree/main/03-steering) (MIT). The data in `batched_steering/data/` is ported from it, except `refusal_prompts.py`, which was written here.

```
batched_steering/   device, model, directions, steer (BatchedSteer/SingleSteer), sweep, generate, loop, score, io
experiments/        m0_hello, m1_repro_sad, m2_batched_vs_loop        (milestones)
                    e1_layer_grid, e2_tomato_map, e3_removal_search  (must-ship sweeps)
                    bench_abc  (A loop / B Strict / C DeferAndFuse / D --compile)
                    e4_refusal, e5_mix                               (stretch)
scripts/            run_vm.sh (TPU VM runner), fig_*.py (figures), check_mix_compile.py, bf16_*.py (bf16 bug repros)
notebooks/          colab_tpu_run.ipynb, self-contained Colab version of every step (built by scripts/build_notebook.py)
tests/              CPU equivalence tests (tiny random Qwen3)
docs/LOG.md         what was run, what broke, the numbers
results/, figures/  JSON + samples + plots from the TPU runs
```

| Step | Result (details in docs/LOG.md) |
|---|---|
| M1 | reference sad trend reproduced (mood +0.53 → −0.20 at L12 2×) |
| M2 | batched == loop 264/264 (greedy fp32), 4.4× faster |
| E1 | 672-setting layer × strength map; L17 1.5× holds up on test topics |
| E2 | tomato map + 1,000-trial search; the hottest single cell generalises worst |
| E3 | 1,000-trial removal search: lower damage than the 40-trial winner, mild winner's curse |
| Bench | B ≈ C ≈ 15× the loop; D (compiled decode step) 28×, 0 recompiles, not bit-exact |
| E4 | refusal direction: removal takes refusals 44% → 3% (KL 0.005); inducing costs fact accuracy |
| E5 | sad + tomato in one row: tomato survives the sad push, but the sad effect doesn't survive the tomato push |

Local check (CPU):
```
HF_HUB_OFFLINE=1 python -m pytest tests -q
python experiments/m2_batched_vs_loop.py --tiny --device cpu --tokens 8 --topics 3 --batch 16
```
On TPU: `pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu`, then run the experiment scripts (the device is picked automatically, `--eager defer` selects DeferAndFuse), or `bash scripts/run_vm.sh e1 e2 ...` on a TPU VM.
