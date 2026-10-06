#!/usr/bin/env python3
"""Figures for the instance-disambiguation study."""
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
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = {"gpt-5.6-sol": "sol (strong)", "gpt-5-mini": "mini", "gpt-5-nano": "nano (weak)"}
COLOR = {"gpt-5.6-sol": BLUE, "gpt-5-mini": ORANGE, "gpt-5-nano": AQUA}
PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
PLABEL = {"both_cognitive": "both\ncognitive", "one_inert": "one\ninert", "both_inert": "both\ninert"}


def load(p):
    return list(csv.DictReader(open(p)))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def m(rows, k):
    xs = [fl(r[k]) for r in rows if fl(r[k]) is not None]
    return mean(xs) if xs else 0.0


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


# --- Figure 1: the spectrum across the ladder (matched 1lite runs) ------------------------
# three panels: experiment-only recall, precision, surviving false cognates, by placement x model
def fig_spectrum():
    data = {md: load(ROOT / "results" / f"instance_instance_hard_stage1lite_{md}.csv") for md in MODELS}
    panels = [("experiment_only_recall", "experiment-only resolved fraction", (0, 1.05)),
              ("precision", "precision", (0, 1.05)),
              ("surviving_instance_fc", "surviving false cognates", None)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for ax, (key, ylab, ylim) in zip(axes, panels):
        for md in MODELS:
            ys = [m([r for r in data[md] if r["placement"] == pl], key) for pl in PLACEMENTS]
            ax.plot(range(3), ys, "-o", color=COLOR[md], label=LABEL[md])
        ax.set_xticks(range(3)); ax.set_xticklabels([PLABEL[p] for p in PLACEMENTS], fontsize=9)
        ax.set_ylabel(ylab); ax.set_facecolor(SURFACE)
        if ylim:
            ax.set_ylim(*ylim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("The instance spectrum across the model ladder: strong resolves and stays precise; "
                 "weak resolves by brute probing and takes traps", fontsize=11.5, y=1.03)
    save(fig, "fig_instance_spectrum.png")


# --- Figure 2: the resolvability curve (sol budget sweep) ---------------------------------
def fig_budget():
    rows = load(ROOT / "results" / "instance_instance_hard_stage3_gpt-5.6-sol.csv")
    budgets = ["0", "3", "unbounded"]
    blabel = {"0": "0\n(none)", "3": "3\n(bounded)", "unbounded": "unbounded"}
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for pl, color, lbl, mk in (("both_cognitive", BLUE, "both cognitive", "-o"),
                               ("one_inert", ORANGE, "one inert", "-s")):
        ys = [m([r for r in rows if r["placement"] == pl and r["budget"] == b],
                "experiment_only_recall") for b in budgets]
        ax.plot(range(3), ys, mk, color=color, label=lbl, markersize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([blabel[b] for b in budgets], fontsize=9)
    ax.set_xlabel("live-system probe budget (interrogations allowed)", fontsize=9)
    ax.set_ylabel("experiment-only resolved fraction")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Live probing resolves the hardest cases only where a live side remains:\n"
                 "budget-limited at both-cognitive (drives to zero), structural at one-inert (no budget helps)",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=10, loc="center left")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("residual → 0", xy=(2, 1.0), xytext=(1.15, 0.98), fontsize=9, color=BLUE)
    ax.annotate("no budget helps", xy=(2, 0.08), xytext=(1.05, 0.16), fontsize=9, color=ORANGE)
    save(fig, "fig_instance_budget.png")


if __name__ == "__main__":
    fig_spectrum()
    fig_budget()
