#!/usr/bin/env python3
"""Six-model capability gradient under TWO-AGENT negotiation (master Figure 8).

Reads results/two_agent_ladder.csv (two-agent, both-cognitive, no reference, the four schema cases,
two trials) and plots, per model, mean precision / resolved fraction / surviving false cognates.
Reader-facing labels say "resolved fraction". Light background, no em-dashes.

The finding this figure now carries: two-agent negotiation COMPRESSES the gradient a lone
reconstructing agent would show. Bilateral ratification holds precision high and drives surviving
false cognates to zero across almost the whole ladder; resolution is uniformly lower and non-monotonic,
the weakest models failing by non-convergence rather than by taking the traps.

Writes figures/fig_ladder_gradient.png.
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
CSV = ROOT / "results" / "two_agent_ladder.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

# strong -> weak order, with short display labels
ORDER = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
LABEL = ["5.6\nSol", "5.6\nTerra", "5.6\nLuna", "5.4", "5.4\nMini", "5.4\nNano"]


def _means():
    rows = [r for r in csv.DictReader(open(CSV)) if r.get("stack", "two-agent") == "two-agent"]
    agg = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r)
    def m(md, k):
        v = [float(x[k]) for x in agg.get(md, []) if x.get(k) not in ("", "None", None)]
        return sum(v) / len(v) if v else float("nan")
    P = [m(md, "precision") for md in ORDER]
    RF = [m(md, "resolved_fraction") for md in ORDER]
    FC = [m(md, "surviving_false_cognates") for md in ORDER]
    return P, RF, FC


def main():
    P, RF, FC = _means()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(ORDER))

    axL.plot(x, P, "-o", color=NAVY, lw=2.2, markersize=7, label="precision", zorder=3)
    axL.plot(x, RF, "-s", color=TEAL, lw=2.2, markersize=7, label="resolved fraction", zorder=3)
    axL.set_title("Precision held high; resolution lower and non-monotonic",
                  fontsize=11, color=NAVY)
    axL.set_ylim(0.0, 1.05)
    axL.set_xticks(x); axL.set_xticklabels(LABEL, fontsize=9)
    axL.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.02, 0.02))

    axR.bar(x, FC, 0.62, color=RED, zorder=3)
    axR.set_title("Surviving false cognates near zero across the ladder", fontsize=11, color=NAVY)
    axR.set_ylim(0.0, 1.0)
    axR.set_xticks(x); axR.set_xticklabels(LABEL, fontsize=9)

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    fig.suptitle("Six-model capability gradient under two-agent negotiation "
                 "(no reference, both-cognitive, mean over 4 cases)",
                 fontsize=12.5, color=NAVY, y=1.0)
    fig.text(0.5, -0.02,
             "Bilateral ratification holds precision high and suppresses the traps across the range; "
             "the weakest model alone takes any, and fails by non-convergence, not commission.",
             ha="center", fontsize=9.5, color=ORANGE, style="italic")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ladder_gradient.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_ladder_gradient.png").name,
          "| P=", [round(p, 2) for p in P], "RF=", [round(r, 2) for r in RF],
          "FC=", [round(f, 2) for f in FC])


if __name__ == "__main__":
    main()
