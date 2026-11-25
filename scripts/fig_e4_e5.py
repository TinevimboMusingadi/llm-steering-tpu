"""Figures for E4 (refusal direction) and E5 (mixing sad + tomato).
python scripts/fig_e4_e5.py [results_dir]  -> figures/e4_refusal.png, figures/e5_mix.png"""
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
    files = sorted(f for f in (RES / sub).glob(pat) if "samples" not in f.name)
    return json.loads(files[-1].read_text()) if files else None


e4 = latest("e4_refusal", "e4_*.json")
if e4 and "test" in e4:
    ind, rem, base = e4["induce"], e4["remove"], e4["baseline"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.6, 1.2, 1]})
    Ls = sorted({r["layer"] for r in ind}); Ss = sorted({r["strength"] for r in ind})
    M = np.full((len(Ss), len(Ls)), np.nan)
    for r in ind:
        if r["direction"] == "refusal":
            M[Ss.index(r["strength"]), Ls.index(r["layer"])] = r["refusal"]
    im = ax[0].imshow(M, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax[0].set_xticks(range(len(Ls)), Ls, fontsize=7); ax[0].set_yticks(range(len(Ss)), [f"{s:g}" for s in Ss])
    rnd = max(r["refusal"] for r in ind if r["direction"] == "random")
    ax[0].set(xlabel="layer (push added after this block)", ylabel="strength",
              title=f"Induce: share of HARMLESS requests refused\n(no edit {base['harmless_train']:.2f}; random direction max {rnd:.2f})")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    for dn, c in (("random", GREY), ("refusal", RED)):
        for a, m in ((1.0, "o"), (0.5, "^")):
            rs = sorted((r for r in rem if r["direction"] == dn and r["ablate"] == a), key=lambda r: r["layer"])
            if rs:
                ax[1].plot([r["layer"] for r in rs], [r["refusal"] for r in rs], marker=m, ms=4, color=c,
                           ls="-" if a == 1 else ":", label=f"{dn} direction, remove x{a:g}")
    ax[1].axhline(base["harmful_train"], color="black", ls="--", lw=0.8, label=f"no edit ({base['harmful_train']:.2f})")
    ax[1].set(xlabel="layer the direction is taken from (removed after every block)", ylabel="share refused",
              ylim=(-0.03, 1.03), title="Remove: share of HARMFUL requests refused")
    ax[1].legend(frameon=False, fontsize=7)
    T = e4["test"]
    names = ["refusal", "random", "shuffled"]
    x = np.arange(len(names))
    ax[2].bar(x - 0.2, [T["induce"]["test"][n]["refusal"] for n in names], 0.4, color=RED, label="induce on harmless (want high)")
    ax[2].bar(x + 0.2, [T["remove"]["test"][n]["refusal"] for n in names], 0.4, color=BLUE, label="remove on harmful (want low)")
    ax[2].axhline(base["harmful_test"], color=BLUE, ls="--", lw=0.8)
    ax[2].axhline(base["harmless_test"], color=RED, ls="--", lw=0.8)
    wi, wr = T["induce"]["train"], T["remove"]["train"]
    ax[2].set_xticks(x, [f"{n}\ndirection" for n in names])
    ax[2].set(ylim=(0, 1.05), ylabel="share refused (test prompts)",
              title=f"Winners on new prompts\ninduce L{wi['layer']} x{wi['strength']:g}, remove L{wr['layer']} x{wr['ablate']:g}")
    ax[2].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "e4_refusal.png", dpi=150); plt.close(fig)
    print("wrote figures/e4_refusal.png")

e5 = latest("e5_mix", "e5_*.json")
if e5 and "test" in e5:
    g = e5["test"]["grid"]
    pairs = list(dict.fromkeys(r["pair"] for r in g))
    S = sorted({r["sad"] for r in g}); Tm = sorted({r["tomato"] for r in g})
    fig, ax = plt.subplots(2, len(pairs) + 1, figsize=(5 * (len(pairs) + 1), 8.2),
                           gridspec_kw={"width_ratios": [1] * len(pairs) + [0.9]})
    for j, pr in enumerate(pairs):
        for i, (key, cmap, lim) in enumerate((("mood", "RdBu", (-1, 1)), ("tomato_success", "viridis", (0, 1)))):
            M = np.full((len(S), len(Tm)), np.nan)
            for r in g:
                if r["pair"] == pr:
                    M[S.index(r["sad"]), Tm.index(r["tomato"])] = r[key]
            im = ax[i, j].imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
            for a in range(len(S)):
                for b in range(len(Tm)):
                    ax[i, j].text(b, a, f"{M[a, b]:+.2f}" if key == "mood" else f"{M[a, b]:.2f}", ha="center",
                                  va="center", fontsize=7, color="black" if key == "mood" else "white")
            ax[i, j].set_xticks(range(len(Tm)), [f"{t:g}" for t in Tm]); ax[i, j].set_yticks(range(len(S)), [f"{s:g}" for s in S])
            ax[i, j].set(xlabel="tomato strength (generated tokens only)", ylabel="sad strength (all tokens)",
                         title=f"{pr}\n{'mood (+1 happy, -1 sad)' if key == 'mood' else 'tomato success'}")
            fig.colorbar(im, ax=ax[i, j], fraction=0.04)
    for i, key in enumerate(("mood_interaction", "tomato_success_interaction")):
        for pr, c in zip(pairs, (RED, BLUE, GOLD)):
            rs = [r for r in g if r["pair"] == pr and r["sad"] > 0 and r["tomato"] > 0]
            ax[i, -1].scatter([r["sad"] for r in rs], [r[key] for r in rs], color=c, s=[8 + 3 * r["tomato"] for r in rs],
                              alpha=0.7, label=pr)
        ax[i, -1].axhline(0, color="black", lw=0.8)
        ax[i, -1].set(xlabel="sad strength (dot size = tomato strength)", ylabel="mixed - (sad alone + tomato alone)",
                      title=f"Interaction: {key.replace('_interaction', '').replace('_', ' ')}\n0 = the two pushes just add up")
        ax[i, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "e5_mix.png", dpi=150); plt.close(fig)
    print("wrote figures/e5_mix.png")
