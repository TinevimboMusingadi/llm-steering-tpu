"""Build notebooks/colab_tpu_run.ipynb: one self-contained Colab notebook that writes the lab code to /content/lab
(%%writefile cells, so no repo access is needed), installs TorchTPU, runs M0 -> M1 -> M2 -> E1 and zips the results.
Re-run this script after changing the package:  python scripts/build_notebook.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([*ROOT.glob("batched_steering/**/*.py"), ROOT / "batched_steering/data/LICENSE-learn-mech-interp",
                *ROOT.glob("experiments/*.py")])


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [md("""# Batched LLM Steering on TPU: lab run

**Before running:** *Runtime → Change runtime type → **v5e-1 TPU*** (or v6e-1). Then *Runtime → Run all*.

The notebook will:
1. install TorchTPU and a recent transformers
2. write the lab code to `/content/lab`
3. run each experiment

| Step | What it does |
|---|---|
| M0 | hello TPU + hooks |
| M1 | reproduce the sad direction, loop vs batched |
| M2 | batched == loop check + speed, Strict vs DeferAndFuse |
| E1 | full layer × strength grid (+ test-topic check) |
| E2 | tomato map + 1,000-trial search |
| E3 | removal search, 1,000 trials, winner's curse |
| Bench | A loop / B Strict / C DeferAndFuse |

At the end it downloads `results.zip`. Drop that zip into the lab repo folder.
Answer key and data: [avbiswas/learn-mech-interp](https://github.com/avbiswas/learn-mech-interp) (MIT)."""),
         code("""# 1. Install (TorchTPU replaces the old torch_xla stack)
!pip -q uninstall -y torch_xla 2>/dev/null
!pip -q install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
!pip -q install -U "transformers>=5.13" accelerate
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch, torch_tpu, transformers
print("torch", torch.__version__, "| torch_tpu", getattr(torch_tpu, "__version__", "?"), "| transformers", transformers.__version__)
print("TPU devices:", torch.accelerator.device_count())
x = torch.randn(4, 4, device="tpu"); print("matmul on tpu ok:", (x @ x).sum().item() is not None)"""),
         md("## 2. Write the lab code")]

for f in FILES:
    rel = f.relative_to(ROOT).as_posix()
    cells.append(code(f"%%writefile /content/lab/{rel}\n" + f.read_text(encoding="utf-8")))
cells.insert(3, code("import os\nfor d in ['batched_steering/data', 'experiments', 'results']:\n"
                     "    os.makedirs(f'/content/lab/{d}', exist_ok=True)"))

RUN = "%cd /content/lab\n!STEER_RESULTS=/content/lab/results python -u experiments/"
cells += [
    md("## 3. M0: hello TPU + hooks\nDone when: zero push == no hook, and a real push changes the text."),
    code(RUN + "m0_hello.py --tokens 48"),
    md("## 4. M1: reproduce the sad direction (fp32, temperature 0.5)\nReference: mood +0.57 (no push) → −0.15 (2×) at layer 12; the random control stays ≈ +0.50.\n"
       "First one config at a time (mode A), then batched (mode B)."),
    code(RUN + "m1_repro_sad.py --mode loop --no-layer-sweep"),
    code(RUN + "m1_repro_sad.py --mode batched --batch 256"),
    md("## 5. M2: batched == loop (greedy, fp32) and speed: Strict vs DeferAndFuse"),
    code(RUN + "m2_batched_vs_loop.py --batch 256 --eager strict"),
    code(RUN + "m2_batched_vs_loop.py --batch 256 --eager defer --tag defer"),
    md("## 6. E1: the full grid (28 layers × 8 strengths × 3 directions = 672 configs, fp32: bf16 batches >128 hit a TorchTPU issue, see LOG)\n"
       "Checkpoints after each direction. If the runtime drops, rerun this cell and it resumes."),
    code(RUN + "e1_layer_grid.py --dtype float32 --eager strict --batch 256"),
    md("## 7. E1 winner on the test topics (winner's curse check)"),
    code(RUN + "e1_layer_grid.py --topic-set test --layers 8,12,14,16,17,18,20 --strengths 1,1.5,2,3 --tag test"),
    md("## 8. E2: tomato map (every layer x 12 strengths, decode-only push) + 1,000-trial search"),
    code(RUN + "e2_tomato_map.py --batch 256"),
    md("## 9. E3: removal random search, 1,000 trials, tune -> test"),
    code(RUN + "e3_removal_search.py --batch 256"),
    md("## 10. Benchmark: A loop vs B batched Strict vs C batched DeferAndFuse"),
    code(RUN + "bench_abc.py --eager strict"),
    code(RUN + "bench_abc.py --eager defer"),
    md("## 11. Download results"),
    code("!cd /content/lab && zip -qr /content/results.zip results\nfrom google.colab import files\nfiles.download('/content/results.zip')"),
]

nb = {"cells": cells, "metadata": {"accelerator": "TPU", "colab": {"provenance": []},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT / "notebooks" / "colab_tpu_run.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells, {len(FILES)} embedded files)")
