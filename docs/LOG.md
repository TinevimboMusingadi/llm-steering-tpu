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
