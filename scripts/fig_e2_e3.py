"""Figures for E2 (tomato map) and E3 (removal search + winner's curse).
python scripts/fig_e2_e3.py [results_dir]  -> figures/e2_tomato_map.png, figures/e3_removal_search.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, RED, GREY, GOLD = "#2a6fdb", "#d1495b", "#9aa0a6", "#e8a33d"


def latest(sub, pat):
    files = sorted((RES / sub).glob(pat))
    files = [f for f in files if "samples" not in f.name]
    return json.loads(files[-1].read_text()) if files else None


e2 = latest("e2_tomato_map", "e2_*.json")
if e2:
    rows = e2["map"]
    Ls = sorted({r["layer"] for r in rows}); Ss = sorted({r["strength"] for r in rows})
    M = np.full((len(Ss), len(Ls)), np.nan)
    for r in rows:
        M[Ss.index(r["strength"]), Ls.index(r["layer"])] = r["success"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(0.01, np.nanmax(M)))
    ax[0].set_xticks(range(len(Ls)), Ls, fontsize=7); ax[0].set_yticks(range(len(Ss)), [f"{s:g}" for s in Ss])
    ax[0].set(xlabel="layer (push added after this block, decode steps only)", ylabel="strength",
              title=f"Tomato success: on topic AND tomatoes AND not repeating ({len(rows)} settings, 36 topics)")
    b = max(rows, key=lambda r: r["success"])
    ax[0].scatter([Ls.index(b["layer"])], [Ss.index(b["strength"])], marker="*", s=250, color=RED, edgecolors="white")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    t = e2["test"]
    names = list(t)
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [t[k]["choose_success"] for k in names], 0.4, color=GREY, label="topics it was chosen on")
    ax[1].bar(x + 0.2, [t[k]["test_success"] for k in names], 0.4, color=BLUE, label="new test topics")
    ax[1].set_xticks(x, [k.replace(" (", "\n(").replace(", ", ",\n") for k in names], fontsize=7)
    ax[1].set(ylim=(0, 1), ylabel="success", title="Chosen vs new topics")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "e2_tomato_map.png", dpi=150); plt.close(fig)
    print("wrote figures/e2_tomato_map.png")

e3 = latest("e3_removal_search", "e3_*.json")
if e3:
    real, rand = e3["search"]["sad"], e3["search"]["random"]
    T = e3["test"]
    thr = e3["args"]["threshold"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].scatter([r["kl"] for r in rand], [r["mood"] for r in rand], s=8, color=GREY, alpha=0.5, label="random direction (control)")
    ax[0].scatter([r["kl"] for r in real], [r["mood"] for r in real], s=8, color=RED, alpha=0.5, label="sad direction")
    for k, m, c in (("winner_40", "s", GOLD), ("winner_all", "*", BLUE)):
        w = T[k]["tune"]
        ax[0].scatter([w["kl"]], [w["mood"]], marker=m, s=200, color=c, edgecolors="black", zorder=3,
                      label=f"winner of {'first 40' if k == 'winner_40' else str(len(real))} trials")
    ax[0].axhline(thr, color="black", ls="--", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=0.01); ax[0].set_xlim(left=0)
    ax[0].set(xlabel="damage: KL on 40 facts (log)", ylabel='mood when asked for "two sad sentences"', ylim=(-1, 1),
              title=f"{len(real)} random removal settings (tune topics)")
    ax[0].legend(frameon=True, fontsize=8, loc="upper left", markerscale=0.6)
    rr = [r for r in T["rows"] if r["tune"]]
    ax[1].scatter([r["tune"]["mood"] for r in rr], [r["test"]["mood"] for r in rr], color=RED, s=25, label="top trials re-run")
    for k, m, c in (("winner_40", "s", GOLD), ("winner_all", "*", BLUE)):
        ax[1].scatter([T[k]["tune"]["mood"]], [T[k]["test"]["mood"]], marker=m, s=200, color=c, edgecolors="black", zorder=3)
    ax[1].axhline(T["hand_picked"]["mood"], color=GREY, ls=":", label=f"hand-picked edit on test ({T['hand_picked']['mood']:+.2f})")
    lo = min(-0.2, min(r["test"]["mood"] for r in rr) - 0.05)
    ax[1].plot([lo, 1], [lo, 1], color="black", lw=0.6, ls="--", label="test = tune")
    ax[1].set(xlabel="mood on tune topics (what the search saw)", ylabel="mood on new test topics",
              title="Winner's curse: do the best trials hold up?", xlim=(lo, 1), ylim=(lo, 1))
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "e3_removal_search.png", dpi=150); plt.close(fig)
    print("wrote figures/e3_removal_search.png")
