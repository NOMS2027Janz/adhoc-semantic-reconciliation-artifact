#!/usr/bin/env python3
"""Construct-cost figure (master Figure 15), TWO-AGENT.

When is building the shared reference worth the cognition it costs? The strong agent's close by
condition across three schema settings, with the cognition spent annotated on each bar. Construction is
load-bearing only where no standard exists (cross-domain, 0.20 -> 0.80); where a given reference already
exists it matches the close for far less cognition.

Reads results/two_agent_full.csv (no-reference floor, both-cognitive, sol) and
results/experiment_closer.csv (constructed and given-reference, two-agent, sol, 2 trials).
Writes figures/fig_construct_cost.png (light background). Single panel: the mini/nano construction-cost
ladder is not re-run two-agent; the master text carries that point via the six-model ladder.
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
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b8c2cc"
NAVY, SURFACE, INK, FAINT = "#14314f", "#fbfaf8", "#1a1a1a", "#6a655e"
SETTINGS = [("config_big_hard", "Scenario 1\nconfiguration\n(a standard exists)"),
            ("config_cross_domain", "Scenario 3\nstandard-free\n(no standard)"),
            ("config_observability", "Scenario 4\nobservability\n(RFC anchor)")]


def _mean(rs, k):
    v = [float(x[k]) for x in rs if x.get(k) not in ("", "None", None)]
    return sum(v) / len(v) if v else 0.0


def _data():
    full = [r for r in csv.DictReader(open(ROOT / "results" / "two_agent_full.csv"))
            if r["stack"] == "two-agent" and r["model"] == "gpt-5.6-sol" and r["uses_reference"] == "False"]
    fa = defaultdict(list)
    for r in full:
        fa[r["case"]].append(r)
    clo = list(csv.DictReader(open(ROOT / "results" / "experiment_closer.csv")))
    ca = defaultdict(list)
    for r in clo:
        ca[(r["case"], r["condition"])].append(r)
    out = {}
    for case, _ in SETTINGS:
        nr = fa.get(case, [])
        co = ca.get((case, "construct-only"), [])
        rf = ca.get((case, "reference"), [])
        out[case] = {
            "no-ref": (_mean(nr, "resolved_fraction"), _mean(nr, "reasoning_tokens")),
            "constructed": (_mean(co, "resolved_fraction"),
                            _mean(co, "construct_tokens") + _mean(co, "reasoning_tokens")),
            "given-ref": (_mean(rf, "resolved_fraction"), _mean(rf, "reasoning_tokens")),
        }
    return out


def main():
    d = _data()
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    x = np.arange(len(SETTINGS)); w = 0.26
    conds = [("no-ref", "no reference", GREY), ("constructed", "constructed", ORANGE),
             ("given-ref", "given reference", BLUE)]
    for i, (key, label, color) in enumerate(conds):
        xs = x + (i - 1) * w
        recs = [d[c][key][0] for c, _ in SETTINGS]
        spends = [d[c][key][1] for c, _ in SETTINGS]
        ax.bar(xs, recs, w, color=color, label=label, zorder=3)
        for xi, r, s in zip(xs, recs, spends):
            sp = f"{s / 1000:.1f}k" if s >= 1000 else f"{s:.0f}"
            ax.annotate(f"{r:.2f}\n{sp} tok", (xi, r), textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=7.4, color=INK,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_title("The strong agent (two agents negotiating): resolved fraction by condition,\n"
                 "with the cognition spent — construction is load-bearing only where no standard exists\n"
                 "(Scenario 3, 0.20 to 0.80); elsewhere a given reference matches it far more cheaply",
                 fontsize=10.5, color=NAVY)
    ax.set_ylabel("resolved fraction  (share of true matches found)", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([s[1] for s in SETTINGS], fontsize=9)
    ax.set_ylim(0, 1.24); ax.axhline(1.0, color=NAVY, ls=":", lw=1)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", facecolor=SURFACE,
              fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.005))
    ax.annotate("no standard —\nbuilding it pays\n(0.20 to 0.80)", xy=(1, 0.80),
                xytext=(1.42, 0.62), fontsize=8.4, color=ORANGE, fontweight="bold",
                ha="left", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("When is constructing the shared reference worth the cognition it costs?",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_construct_cost.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_construct_cost.png").name)


if __name__ == "__main__":
    main()
