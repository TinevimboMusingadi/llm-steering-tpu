# Lab log: what broke, what fixed it (blog fuel)

## 2026-11-22: P0, core package on CPU
- **Hidden states ignore hooks (transformers 5.13).** `output_hidden_states=True` records each block's output *before* user forward hooks run. A steering push therefore doesn't show up in `hidden_states[L]`. It only shows up downstream, or through a read hook registered after the steer hook. Directions are unaffected because they're computed with no hooks attached. Tests measure edits with read hooks.
- **Any return value from a forward hook replaces the block output.** A read-only lambda like `lambda m,i,o: seen.setdefault(...)` silently swapped the hidden state for a tuple. Read-only hooks must return `None`.
- **StaticCache in v5 tracks its own write position** (`cumulative_length` tensor; no `cache_position` arg anymore). Passing a fixed `[B, T+N]` attention mask plus explicit `position_ids` (left padding!) gives identical greedy tokens to `model.generate`. Tested.
- **Two steering objects on one model:** hooks from an idle `BatchedSteer` fired during the loop runner (batch-size mismatch crash). `Sweep` now enables its hooks only inside its own calls.
- **HF Xet downloader hung** on this Windows box. `HF_HUB_DISABLE_XET=1` avoids the hang. Big downloads are still very slow locally, so real-model runs happen on Colab TPU and local runs use a tiny random 28-layer Qwen3 (`--tiny`).
- **Results (CPU, tiny model, plumbing only):**
  - 7/7 tests pass: batched == one-at-a-time (push, negative push, decode-only push, removal), dummy padding rows don't matter, fact damage matches the loop, and buffers are reused.
  - M2 smoke: 33/33 rows match, batched 2.2x faster even on CPU.

## 2026-11-22: first TPU runs (spot v5e-1, us-east1-c, torch 2.15 + torch_tpu, transformers 5.24)
- **M0 passed.** Forward hooks work on TorchTPU: zero push == no hook, and a push changes the text.
- **M1 reproduced** the reference trend:
  - mood 0.59 with no push → −0.05 at 2× in the loop run, 0.53 → −0.20 in the batched run (reference 0.57 → −0.15);
  - facts 0.85 → 0.65, KL 0.361 (reference 0.358).
- **Batched speed-up:** 4.4× in M2 (11 configs × 24 prompts). E1 runs 2.9 configs/s against 0.2 for the loop.
- **Bug #1 (ours): `h @ r` vs `(h*r).sum(-1)`.** On TPU an fp32 matmul runs at bf16 input precision by default. The loop's removal used a matmul and drifted from the batched hook over 28 blocks (5/264 rows). Both paths now use elementwise multiply + sum, and M2 fp32 gives 264/264 identical.
- **Bug #2 (platform, bf16): rows ≥ 128 corrupted.**
  - Symptom: with bf16 activations, a steered batch of 256 whose rows have different left padding produces garbage on rows 128–255 ("Winter! Here are a sentence about winter dog"). Rows 0–127 are exact.
  - All of these are clean: fp32 at 256, bf16 at 128, unsteered bf16 at 256, and steered bf16 at 256 when every row has the same prompt.
  - Changing the hook math (bf16-native, or add-in-fp32-then-cast) does NOT fix it.
  - Workaround: `Sweep` caps bf16 TPU batches at 128.
  - Impact: the first E1 run (bf16, B=256) is half corrupted and must be rerun.
  - Repro scripts: `scripts/bf16_*.py`. To file upstream with a minimal repro.
- **DeferAndFuse** was no faster than Strict for these workloads (M2 13.4 s vs 14.0 s; M0 slightly slower). Most of the speed comes from batching.

## 2026-11-23: E1 test check, E2, E3, bench (second spot v5e-1, us-east1-c, fp32 Strict, batch 256)
- **E1 winner on the test topics (winner's curse check).** L17 1.5× goes from −0.32 (tune) to **−0.23 (test)**, KL 0.23. The reference's L12 2× gives −0.09 on test (KL 0.36). L14 2× gives −0.24 at KL 0.27. Random control ≥ +0.46, shuffled ≥ +0.13.
- **E2 tomato map: 336 settings (28 layers × 12 strengths) + 1,000 random settings, all on 36 topics.** That is 48k replies in 18 min, about 1.25 configs/s (36 prompts per config).
  - L18 reproduces the reference: 0 / 0.44 / 0.17 / 0.03 at 5/10/15/20× (reference 0 / 0.44 / 0.25 / 0.03).
  - The window is narrow and diagonal: early layers need ~12–14×, L18–20 ~10×, and L25–27 only 4–8×. It disappears after L21 except for a lone hot spot at L26 6×.
  - Chosen → test: by hand (L18 10×) 0.44 → 0.33; map best (L26 6×) 0.53 → **0.25**; 60-trial search 0.42 → 0.25 (reference 0.47 → 0.21); **1,000-trial search (L17 12.3×, spread 2) 0.56 → 0.42**. The single hottest cell is the least reliable. The wider search finds settings that generalise better.
- **E3 removal search: 1,000 trials + 1,000 random-direction trials, tune → test.** That is 24k replies in ~18 min, 3.2 configs/s.
  - The same RNG stream as the reference: our trial #2 *is* the reference winner (dir L13, 1.17×, blocks 12–19), KL 0.045 vs 0.042. Its tune mood here is 0.70 vs 0.87 (sampling noise; one sample per topic).
  - 40-trial winner: dir L14 0.67× blocks 4–16, mood 0.90, KL 0.059 → test 0.90.
  - 1,000-trial winner: dir L13 0.36× blocks 6–25, mood 0.82, **KL 0.042** → test **0.87**. About 30% less damage at the same effect.
  - 91 of 1,000 trials pass the 0.8 bar (2 of the first 40). The random-direction search never gets above +0.01.
  - Winner's curse is mild for removal: the top 40 drop 0.05 on average from tune to test, with a few outliers dropping to 0.44–0.6.
  - Hand-picked (L12, full, every block) on test: +0.34 (reference 0.353).
- **Bench A/B/C (E1-style pushes, 12 prompts × 60 tokens, fp32):**
  - A loop: 0.21–0.22 configs/s.
  - B Strict: 1.20 (8 configs) → 2.49 (32) → **3.36 (64)** → 3.28 (128) → 3.08 (224) configs/s.
  - C DeferAndFuse: 1.28 → 2.48 → 3.22 → 3.21 → 3.13.
  - The speed-up saturates at ~15× once a batch is full (256 rows = 21 configs × 12 prompts); B and C are within noise.

## 2026-11-24/25: stretch round, E4 refusal, E5 mixing, mode D (third spot v5e-1, us-east1-c, fp32 Strict)
- **Core: second push slot.** `BatchedSteer` gained `vec2/strength2/layer_mask2/prompt_scale2`, set from `Config.extra=[Config(...)]`. One row can now hold sad on every token at one block plus tomato on generated tokens at others. The removal term now sees the pushed state (h + pushes), the same order as `SingleSteer`'s hooks. Old configs give identical tensors. New CPU test: 8/8 pass. On TPU (greedy fp32, `scripts/check_mix_compile.py`): mixed rows batched == loop **29/30**, and the one miss splits at char 104 of 110 (late fp32 drift, as in M2).
- **Gotcha: CRLF.** Python `Path.write_text` on Windows wrote `run_vm.sh` with CRLF and bash on the VM refused it (`$'do\r'`). All edited files were converted back to LF.
- **E4 refusal direction** (harmful − harmless mean at the last prompt token, 64 + 64 hand-written requests, half to choose / half to test; string-match refusal score on the first 160 chars):
  - Baseline: explicit refusals on harmful requests are only 0.38 (train) / 0.44 (test), 0.00 on harmless. Qwen3-0.6B often just complies, and soft "this is a serious violation…" lectures don't count as refusals.
  - Induce (push on harmless, 28 layers × 5 strengths): a band at L16–20. Peak **0.91 at L20 2×**, but facts 0.85 → 0.13, KL 2.9. With KL < 0.2 the best is L12 3×: 0.22 train → 0.34 test. Random/shuffled at the same setting: 0.00. Random max over the whole map: 0.12.
  - Remove (project out the layer-L direction after every block, ×0.5 / ×1): taking the direction from L ≥ 14 takes refusals to ~0. Winner L25 ×0.5: 0.00 train, KL 0.005 → **0.03 on test** (random 0.22, shuffled 0.28, no edit 0.44).
  - 392 configs, 12.5k replies, ~4.5 min including fact damage.
  - Ablated harmful replies are saved as 60 chars only.
- **E5 mixing sad + tomato** (two pairings × 5 × 5 strengths + 3 sanity rows, choose 36 / test 24 topics):
  - cos(sad, tomato) ≈ 0 (+0.02 at L12, +0.04 at L18).
  - Tomato survives sad (tomato interaction ≈ 0: −0.04 / +0.08).
  - **Sad does not survive tomato.** Test L12+L18: sad 2× alone −0.16, tomato 10× alone +0.53, both **+0.27**. The mean mood interaction is +0.24 / +0.20 (mixed is happier than additive). Possible causes: the topic shifts to food, or the larger tomato push dominates the residual stream; equal-norm pushes would separate them.
  - Sanity: sad 2× + removal at the same block = +0.57 (no edit +0.55, push alone −0.17), so it cancels.
- **Mode D: decode step under `torch.compile(backend="tpu", dynamic=False)`.** It needed `cache_position` passed as a slice of a preallocated device tensor (no Python-int counter).
  - Warm-up 28 s, 1 graph, then **0 new graphs** over 5 sweep sizes.
  - Peak **5.72 configs/s, 1.7× mode B, ~28× the loop**. Unlike DeferAndFuse, compile pays off.
  - **Not bit-exact:** compiled vs eager greedy **24/30** identical on heavily steered rows, and splits start as early as char 5 (different fused arithmetic flips near-tied argmaxes). Use B for exact equivalence checks and D for throughput.
- Results backed up to the private GCS bucket (run-2026-11-24). VM deleted.
