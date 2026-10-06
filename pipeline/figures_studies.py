#!/usr/bin/env python3
"""Figures for the reference-anatomy and pre-lift baseline studies."""
from __future__ import annotations
import csv, sys
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

def load(p): return list(csv.DictReader(open(p)))
def fl(x):
    try: return float(x)
    except: return None
def m(rows, k):
    xs = [fl(r[k]) for r in rows if fl(r[k]) is not None]
    return mean(xs) if xs else 0.0

def save(fig, name):
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)

# ---------------- Figure 1: reference-field ablation (opaque), trap survival by field x model
def fig_anatomy():
    rows = load(ROOT / "results" / "ablation_config_big_hard_opaque.csv")
    inert = [r for r in rows if r["placement"] in ("one_inert", "both_inert")]
    cells = [("no-reference", None), ("id-only", "id-only"), ("+class", "class"),
             ("+definition", "definition"), ("+example", "example"),
             ("+def+example", "definition+example"),
             ("full", "lexical+class+definition+example")]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    n = len(cells); w = 0.26
    for i, model in enumerate(MODELS):
        vals = []
        for _, cell in cells:
            if cell is None:
                sel = [r for r in inert if r["model"] == model and r["uses_reference"] == "False"]
            else:
                sel = [r for r in inert if r["model"] == model and r["uses_reference"] == "True" and r["cell"] == cell]
            vals.append(m(sel, "surviving_false_cognates"))
        xs = [j + (i - 1) * w for j in range(n)]
        ax.bar(xs, vals, width=w, color=COLOR[model], label=LABEL[model])
    ax.set_xticks(range(n)); ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.set_ylabel("surviving false cognates (mean)")
    ax.set_title("Reference fields help most where cognition is weakest — and $class$ hurts the weak model",
                 fontsize=11)
    ax.axhline(0, color="#999", lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig_anatomy_fields.png")

# ---------------- Figure 2: pre-lift lift ladder — recall and trap survival side by side
def fig_lift():
    rows = load(ROOT / "results" / "lift_baseline_config_big_hard.csv")
    ladder = [("lexical\nonly", set()),
              ("+expl", {"explanation"}),
              ("+class", {"explanation", "class"}),
              ("+struct", {"explanation", "class", "structure"}),
              ("+inst\n(full)", {"explanation", "class", "structure", "instances"})]
    def cellset(r): return {f for f in ("explanation", "class", "structure", "instances") if r[f] == "1"}
    fig, (axr, axf) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for model in MODELS:
        rec, fc = [], []
        for _, on in ladder:
            sel = [r for r in rows if r["model"] == model and cellset(r) == on]
            rec.append(m(sel, "recall")); fc.append(m(sel, "surviving_false_cognates"))
        axr.plot(range(len(ladder)), rec, "-o", color=COLOR[model], label=LABEL[model])
        axf.plot(range(len(ladder)), fc, "-o", color=COLOR[model], label=LABEL[model])
    for ax, ttl, yl in ((axr, "Resolved fraction rises with the lift, for every model", "resolved fraction"),
                        (axf, "But the weak model also takes more traps", "surviving false cognates")):
        ax.set_xticks(range(len(ladder))); ax.set_xticklabels([l[0] for l in ladder], fontsize=8.5)
        ax.set_title(ttl, fontsize=10.5); ax.set_ylabel(yl); ax.set_facecolor(SURFACE)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axr.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("The lift carries the distinction — but exploiting it is capability-gated",
                 fontsize=11.5, y=1.02)
    save(fig, "fig_lift_ladder.png")

if __name__ == "__main__":
    fig_anatomy()
    fig_lift()
