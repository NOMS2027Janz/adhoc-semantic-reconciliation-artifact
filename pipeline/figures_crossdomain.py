#!/usr/bin/env python3
"""Figures for the cross-domain (standard-free) study. The seam schematic is data-
independent (it draws the setting); the results figures read the wave CSVs."""
from __future__ import annotations
import csv, glob
import numpy as np
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, SURFACE = "#9a958d", "#fbfaf8"
NAVY, INK = "#14314f", "#1a1a1a"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


def fig_seam():
    """The scene: two bespoke models, adjacent domains, meeting at one seam."""
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_facecolor(SURFACE)

    # domain headers
    ax.add_patch(FancyBboxPatch((0.3, 8.2), 3.7, 1.1, boxstyle="round,pad=0.08",
                                fc="#eaf1fb", ec=BLUE, lw=1.5))
    ax.text(2.15, 8.75, "Agent X — Meridian", ha="center", va="center", fontsize=11.5,
            weight="bold", color="#14314f")
    ax.text(2.15, 8.4, "bespoke transport OSS", ha="center", va="center", fontsize=9, color="#555")
    ax.add_patch(FancyBboxPatch((6.0, 8.2), 3.7, 1.1, boxstyle="round,pad=0.08",
                                fc="#eafbf3", ec=AQUA, lw=1.5))
    ax.text(7.85, 8.75, "Agent Y — Cascade", ha="center", va="center", fontsize=11.5,
            weight="bold", color="#14314f")
    ax.text(7.85, 8.4, "bespoke IP/VPN controller", ha="center", va="center", fontsize=9, color="#555")

    # the five seam correspondences (aligned rows), the grade false cognate, then native gaps
    rows = [
        ("circuit", "underlay", "seam: the underlay", "corr"),
        ("hand-off", "attachment", "one demarcation", "corr"),
        ("rate", "rate", "committed payload, not line rate", "corr"),
        ("latency", "latency", "a bound: ms  ==  tier", "corr"),
        ("protection", "protection", "against a path failure", "corr"),
        ("grade", "grade", "UNRELATED: transport class  vs  IP class", "false"),
    ]
    y = 7.3
    for a, b, note, kind in rows:
        col = ORANGE if kind == "false" else AQUA
        ax.add_patch(FancyBboxPatch((0.5, y - 0.28), 3.0, 0.56, boxstyle="round,pad=0.04",
                                    fc="white", ec=BLUE, lw=1.1))
        ax.text(2.0, y, a, ha="center", va="center", fontsize=10, color="#14314f")
        ax.add_patch(FancyBboxPatch((6.5, y - 0.28), 3.0, 0.56, boxstyle="round,pad=0.04",
                                    fc="white", ec=AQUA, lw=1.1))
        ax.text(8.0, y, b, ha="center", va="center", fontsize=10, color="#14314f")
        if kind == "false":
            ax.plot([3.5, 6.5], [y, y], color=col, lw=1.6, ls=(0, (4, 3)))
            ax.text(5.0, y + 0.16, "false cognate", ha="center", va="bottom", fontsize=8,
                    color=col, weight="bold")
            ax.text(5.0, y - 0.2, "✗", ha="center", va="center", fontsize=13, color=col)
        else:
            ax.plot([3.5, 6.5], [y, y], color=col, lw=1.8)
            ax.text(5.0, y + 0.14, note, ha="center", va="bottom", fontsize=7.6, color="#555")
        if kind == "false":
            ax.text(5.0, y - 0.36, note, ha="center", va="top", fontsize=7.2, color=col)
        y -= 0.92

    # native gaps
    ax.text(2.0, 1.2, "native only:  bearer / wavelength", ha="center", fontsize=8.5, color=GREY, style="italic")
    ax.text(8.0, 1.2, "native only:  service, vlan", ha="center", fontsize=8.5, color=GREY, style="italic")

    ax.text(5.0, 0.35, "Two home-grown models, no public standard. The worlds overlap only at "
            "the seam; everything else is native to one side.", ha="center", fontsize=9.2, color="#333")
    ax.set_title("The standard-free scenario — two home-grown models meeting where a Cascade "
                 "service rides a Meridian circuit", fontsize=12, y=1.0)
    save(fig, "fig_crossdomain_seam.png")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(pat):
    rows = []
    for f in glob.glob(str(ROOT / "results" / pat)):
        rows += list(csv.DictReader(open(f)))
    return rows


def fig_mirror():
    """Cross-domain at both-cognitive (two agents): resolved fraction without vs with the constructed
    reference, precision annotated. Grouped bars (no positional collisions)."""
    rows = [r for r in load("two_agent_full.csv")
            if r.get("stack") == "two-agent" and r.get("case") == "config_cross_domain"]
    if not rows:
        print("skip fig_crossdomain_mirror: no two_agent_full data"); return
    MODELS = [("gpt-5.6-sol", "sol\n(strong)"), ("gpt-5-mini", "mini\n(mid)"), ("gpt-5-nano", "nano\n(weak)")]

    def cell(md, ref, k):
        xs = [fl(r[k]) for r in rows if r["model"] == md and r["uses_reference"] == ref and fl(r[k]) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(MODELS)); w = 0.36
    rf_no = [cell(m, "False", "resolved_fraction") for m, _ in MODELS]
    rf_re = [cell(m, "True", "resolved_fraction") for m, _ in MODELS]
    p_no = [cell(m, "False", "precision") for m, _ in MODELS]
    p_re = [cell(m, "True", "precision") for m, _ in MODELS]
    b0 = ax.bar(x - w / 2, rf_no, w, color=GREY, label="no shared reference", zorder=3)
    b1 = ax.bar(x + w / 2, rf_re, w, color=BLUE, label="constructed reference", zorder=3)
    for xi, rf, p in zip(x - w / 2, rf_no, p_no):
        ax.annotate(f"precision\n{p:.2f}", (xi, rf), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for xi, rf, p in zip(x + w / 2, rf_re, p_re):
        ax.annotate(f"precision\n{p:.2f}", (xi, rf), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.axhline(1.0, color=NAVY, ls=":", lw=1)
    ax.set_ylabel("resolved fraction  (share of true matches found)", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in MODELS], fontsize=9.5)
    ax.set_ylim(0, 1.22)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", facecolor=SURFACE, fontsize=9,
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Standard-free at both-cognitive (two agents): the mirror.\n"
                 "Without the constructed reference the strong agent under-commits (0.20 resolved, precision\n"
                 "1.00 — omission) while the weak agent commits some wrong (precision 0.67); the reference\n"
                 "carries the strong agent to a full close and restores the weak agent's precision.",
                 fontsize=9.8, color=NAVY)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True)
    save(fig, "fig_crossdomain_mirror.png")


if __name__ == "__main__":
    fig_seam()
    fig_mirror()
