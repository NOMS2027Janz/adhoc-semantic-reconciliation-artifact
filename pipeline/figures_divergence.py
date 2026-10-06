#!/usr/bin/env python3
"""The lexicon-divergence sweep (report 1/4 Figure, master reference).

Reads results/divergence_sweep.csv (reference on, both_inert) and plots, per model across the
divergence axis (share of concept-pairs on an independent lexicon rather than a shared bridge):
LEFT  precision -- flat at 1.0 for the capable agent across the whole sweep (no break);
RIGHT confident errors (mean per run) -- the one quantity divergence moves, and only at the weak end.

The finding: lexicon divergence is not the axis that defeats a capable agent's alignment. Precision
holds and no planted false cognate is taken at any level; what rises with divergence is the weak
model's confident, spurious assertion. Light background, no em-dashes.

Writes figures/fig_divergence.png.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
CSV = ROOT / "results" / "divergence_sweep.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = ["sol (strong)", "mini (mid)", "nano (weak)"]
COLOR = [NAVY, TEAL, ORANGE]
MARK = ["-o", "-s", "-^"]
LEVELS = [0, 2, 4, 6, 8]
NPAIRS = 8


def _means():
    rows = [r for r in csv.DictReader(open(CSV))
            if r.get("uses_reference") == "True" and r.get("placement") == "both_inert"]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], int(r["level_pairs"]))].append(r)

    def m(md, p, k):
        v = [float(x[k]) for x in agg.get((md, p), []) if x.get(k) not in ("", "None", None)]
        return sum(v) / len(v) if v else float("nan")

    prec = {md: [m(md, p, "precision") for p in LEVELS] for md in ORDER}
    cerr = {md: [m(md, p, "confident_errors") for p in LEVELS] for md in ORDER}
    return prec, cerr


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("lexicon divergence  (share of pairs on an independent lexicon)",
                  fontsize=9.5, color=INK)


def main():
    prec, cerr = _means()
    x = [p / NPAIRS for p in LEVELS]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    for md, lab, c, mk in zip(ORDER, LABEL, COLOR, MARK):
        axL.plot(x, prec[md], mk, color=c, lw=2.2, markersize=7, label=lab, zorder=3)
    axL.set_title("Precision holds across the sweep: no break for a capable agent",
                  fontsize=11, color=NAVY)
    axL.set_ylim(0.80, 1.01)
    axL.set_ylabel("precision", fontsize=10, color=INK)
    axL.legend(frameon=False, fontsize=9.5, loc="lower right")
    _style(axL)
    axL.text(0.0, 0.807, "planted false cognates taken: 0, every model, every level",
             fontsize=8.5, color=FAINT, style="italic", ha="left", va="bottom")

    # sol and mini are both zero at every level; draw the strong line a hair above the axis so it is
    # not hidden under the mid line (a visibility offset only, noted in the caption).
    ROFF = {"gpt-5.6-sol": 0.02, "gpt-5-mini": 0.0, "gpt-5-nano": 0.0}
    for md, lab, c, mk in zip(ORDER, LABEL, COLOR, MARK):
        y = [v + ROFF[md] for v in cerr[md]]
        axR.plot(x, y, mk, color=c, lw=2.2, markersize=7, label=lab, zorder=3)
    axR.set_title("What divergence moves: the weak model's confident errors",
                  fontsize=11, color=NAVY)
    axR.set_ylim(-0.03, 1.0)
    axR.set_ylabel("confident errors  (mean per run)", fontsize=10, color=INK)
    axR.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style(axR)

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in x], fontsize=9)

    fig.suptitle("Lexicon-divergence sweep  (reference on, both-inert; four trials per cell)",
                 fontsize=12.5, color=NAVY, y=1.02)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_divergence.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_divergence.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
