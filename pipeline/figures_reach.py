#!/usr/bin/env python3
"""Figure for the reach study: how far an agent's reconciliation reaches as a function of
what it can SEE (visibility) and what it can ASK (interrogation), on the hard instance case
at both_cognitive with the strong model, plus the interrogation x model ladder.

Reads results/reach_instance_hard_both.csv; writes figures/fig_reach.png.
"""
from __future__ import annotations
import csv
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fbfaf8"
CSV = ROOT / "results" / "reach_instance_hard_both.csv"

AFF = ["none", "lookup", "discovery", "full"]
AFF_LABEL = ["none\n(decide from\nrecord)", "lookup\n(named\nattribute)", "discovery\n(ask what\nexists)", "full\n(+ virtual\nexperiment)"]
VIS = ["key", "name+key", "name+key+attrs", "name+key+attrs+rels"]
VIS_LABEL = ["key only", "+ name", "+ attributes", "+ topology"]
VIS_COLOR = ["#c9d9ef", "#8fb4e2", "#4f86d0", "#1c4f92"]   # light -> dark blue
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
MLABEL = {"gpt-5.6-sol": "sol (strong)", "gpt-5-mini": "mini (mid)", "gpt-5-nano": "nano (weak)"}
MCOLOR = {"gpt-5.6-sol": BLUE, "gpt-5-mini": ORANGE, "gpt-5-nano": AQUA}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    return list(csv.DictReader(open(CSV)))


def mean_eo(rows, **sel):
    xs = [fl(r["experiment_only_recall"]) for r in rows
          if all(r[k] == v for k, v in sel.items()) and fl(r["experiment_only_recall"]) is not None]
    return mean(xs) if xs else float("nan")


def main():
    rows = load()
    fig, (axg, axl) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    x = range(len(AFF))

    # --- left: the grid, strong model, one line per visibility level -----------------------
    for vis, lbl, col in zip(VIS, VIS_LABEL, VIS_COLOR):
        ys = [mean_eo(rows, model="gpt-5.6-sol", evidence=vis, interrogation=a) for a in AFF]
        axg.plot(x, ys, "-o", color=col, label=lbl, markersize=6, linewidth=2)
    axg.set_title("The strong agent: what it can SEE x what it can ASK\n"
                  "visibility alone caps at 0.5; discovery is the hinge to a full close",
                  fontsize=11)
    axg.set_xticks(list(x)); axg.set_xticklabels(AFF_LABEL, fontsize=8.5)
    axg.set_xlabel("interrogation (what the agent may ask)", fontsize=9.5)
    axg.set_ylabel("experiment-only resolved fraction")
    axg.set_ylim(-0.05, 1.12)
    axg.legend(title="visibility", frameon=False, fontsize=8.5, title_fontsize=8.5,
               loc="upper left", bbox_to_anchor=(0.02, 1.0))
    axg.annotate("discovery unlocks\na full close", xy=(2, 0.98), xytext=(2.15, 0.55), fontsize=8.5,
                 color=BLUE, ha="left", arrowprops=dict(arrowstyle="->", color=BLUE))

    # --- right: reach x power, full visibility, one line per model -------------------------
    for mdl in MODELS:
        ys = [mean_eo(rows, model=mdl, evidence="name+key+attrs+rels", interrogation=a) for a in AFF]
        axl.plot(x, ys, "-o", color=MCOLOR[mdl], label=MLABEL[mdl], markersize=6, linewidth=2)
    axl.set_title("Reach x power (full visibility)\n"
                  "strong converts reach to a full close; mid plateaus; weak's resolved fraction is commit-happy",
                  fontsize=10)
    axl.set_xticks(list(x)); axl.set_xticklabels(AFF_LABEL, fontsize=8.5)
    axl.set_xlabel("interrogation (what the agent may ask)", fontsize=9.5)
    axl.set_ylim(-0.05, 1.08)
    axl.legend(title="model", frameon=False, fontsize=9, title_fontsize=9, loc="lower right")

    for ax in (axg, axl):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_reach.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_reach.png").name)


if __name__ == "__main__":
    main()
