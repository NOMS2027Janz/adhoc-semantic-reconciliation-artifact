#!/usr/bin/env python3
"""Figures for the verification study (results/verify_verify_hard.csv)."""
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
PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
PLABEL = {"both_cognitive": "both\ncognitive", "one_inert": "one\ninert", "both_inert": "both\ninert"}


def load():
    return list(csv.DictReader(open(ROOT / "results" / "verify_verify_hard.csv")))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def m(rows, k):
    xs = [fl(r[k]) for r in rows if fl(r[k]) is not None]
    return mean(xs) if xs else 0.0


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


# --- Figure 1: what each mode catches, by error category (at both_cognitive, full reach) ---
def fig_modes():
    rows = load()
    modes = [("byte", "byte round-trip"), ("virtual", "virtual operation"),
             ("invariant", "invariant round-trip")]
    # invariant is model-dependent; average over models. byte/virtual are deterministic.
    def cat(mode, key):
        rs = sel(rows, mode=mode, placement="both_cognitive")
        return m(rs, key)
    mv = [cat(md, "mv_catch_rate") for md, _ in modes]
    bc = [cat(md, "bc_catch_rate") for md, _ in modes]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = range(len(modes)); w = 0.38
    ax.bar([i - w/2 for i in x], mv, width=w, color=ORANGE, label="meaning-visible error")
    ax.bar([i + w/2 for i in x], bc, width=w, color=BLUE, label="byte-clean error")
    ax.set_xticks(list(x)); ax.set_xticklabels([lbl for _, lbl in modes], fontsize=10)
    ax.set_ylabel("catch rate (fraction of wrong pairs failed)")
    ax.set_ylim(0, 1.05)
    ax.set_title("The modes are complementary — only together catch every wrong correspondence",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, (a, b) in enumerate(zip(mv, bc)):
        ax.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8, color="#333")
        ax.text(i + w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8, color="#333")
    save(fig, "fig_verify_modes.png")


# --- Figure 2: verification reach collapses across the spectrum ---------------------------
def fig_reach():
    rows = load()
    fig, (axr, axc) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    # left: reach by mode across placements
    for mode, color, lbl in (("byte", "#9aa3ad", "byte"), ("invariant", ORANGE, "invariant"),
                             ("virtual", BLUE, "virtual operation")):
        reach = [m(sel(rows, mode=mode, placement=pl), "reach") for pl in PLACEMENTS]
        axr.plot(range(3), reach, "-o", color=color, label=lbl)
    axr.set_title("Verification reach falls as cognition recedes", fontsize=10.5)
    axr.set_ylabel("reach (fraction of proposals decidable)")
    # right: byte-clean catch by mode across placements — only the virtual op, only at both_cog
    for mode, color, lbl in (("byte", "#9aa3ad", "byte"), ("invariant", ORANGE, "invariant"),
                             ("virtual", BLUE, "virtual operation")):
        bc = [m(sel(rows, mode=mode, placement=pl), "bc_catch_rate") for pl in PLACEMENTS]
        axc.plot(range(3), bc, "-o", color=color, label=lbl)
    axc.set_title("Byte-clean errors are catchable only with full cognition", fontsize=10.5)
    axc.set_ylabel("byte-clean catch rate")
    for ax in (axr, axc):
        ax.set_xticks(range(3)); ax.set_xticklabels([PLABEL[p] for p in PLACEMENTS], fontsize=9)
        ax.set_ylim(-0.03, 1.08); ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axr.legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("The virtual operation carries verification — and its reach is capability-gated",
                 fontsize=11.5, y=1.02)
    save(fig, "fig_verify_reach.png")


if __name__ == "__main__":
    fig_modes()
    fig_reach()
