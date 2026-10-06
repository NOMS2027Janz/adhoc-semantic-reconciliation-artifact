#!/usr/bin/env python3
"""E19 -- the design-time / run-time split (report figure).

Reads results/design_run_split.csv and shows, per model:
  LEFT   partition accuracy -- how well the agent sorts each seam fact into design-time FREEZE vs
         run-time LIVE (gold freeze fraction marked as a reference line).
  RIGHT  the directed errors, mean count per run: LIVE-as-FREEZE (the staleness hazard -- freezing a
         fact that will drift or was never fixed; the confident share hatched) and FREEZE-as-LIVE
         (over-caution -- needless run-time cost).

Light background.

    python figures_design_run_split.py                     # default results CSV
    python figures_design_run_split.py results/e19_x.csv   # explicit CSV (testing)

Writes figures/fig_design_run_split.png.
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
CSV = ROOT / "results" / "design_run_split.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LAB = ["sol\n(strong)", "mini\n(mid)", "nano\n(weak)"]


def _means(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    agg = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r)

    def m(model, k):
        v = [float(x[k]) for x in agg.get(model, []) if x.get(k) not in ("", "None", None)]
        return sum(v) / len(v) if v else np.nan
    return agg, m


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    agg, m = _means(csv_path)
    present = [x for x in ORDER if x in agg]
    if not present:
        raise SystemExit(f"no usable rows in {csv_path}")
    labels = [LAB[ORDER.index(x)] for x in present]
    x = np.arange(len(present))
    gold_freeze = float(agg[present[0]][0]["gold_freeze_fraction"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    w = 0.38
    tacc = [m(md, "timing_accuracy") for md in present]
    lacc = [m(md, "layer_accuracy") for md in present]
    axL.bar(x - w / 2, tacc, width=w, color=TEAL, zorder=3, label="timing (freeze/live)")
    axL.bar(x + w / 2, lacc, width=w, color=NAVY, zorder=3, label="layer (4-way)")
    for xs, vals in ((x - w / 2, tacc), (x + w / 2, lacc)):
        for xi, v in zip(xs, vals):
            if not np.isnan(v):
                axL.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    axL.axhline(gold_freeze, ls="--", color=FAINT, lw=1.2, zorder=2)
    axL.text(len(present) - 0.5, gold_freeze + 0.01, f"gold freeze fraction {gold_freeze:.2f}",
             ha="right", va="bottom", fontsize=8, color=FAINT, style="italic")
    axL.set_title("Sorting the seam facts: timing and layer accuracy", fontsize=11, color=NAVY)
    axL.set_ylabel("fraction of facts classified correctly", fontsize=10, color=INK)
    axL.set_ylim(0, 1.10)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=9.5)
    axL.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(axL)

    laf = [m(md, "live_as_freeze") for md in present]
    lafc = [m(md, "live_as_freeze_confident") for md in present]
    pvp = [m(md, "provenance_as_pragmatic") for md in present]
    # staleness hazard (red; confident portion solid over translucent) and the prov/prag conflation
    axR.bar(x - w / 2, laf, width=w, color=RED, alpha=0.45, zorder=3, label="live→freeze (staleness)")
    axR.bar(x - w / 2, lafc, width=w, color=RED, zorder=4, label="  of which confident")
    axR.bar(x + w / 2, pvp, width=w, color=ORANGE, zorder=3, label="provenance→pragmatic (conflation)")
    for xs, vals in ((x - w / 2, laf), (x + w / 2, pvp)):
        for xi, v in zip(xs, vals):
            if not np.isnan(v) and v > 0.02:
                axR.text(xi, v + 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    axR.set_title("The two hazards: staleness, and the provenance/pragmatic conflation",
                  fontsize=10.5, color=NAVY)
    axR.set_ylabel("facts misclassified (mean per run)", fontsize=10, color=INK)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=9.5)
    axR.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(axR)

    fig.suptitle("The design-time / run-time split across the four layers: freeze the schema and the "
                 "provenance (what a thing is, who to ask); resolve live the instances and pragmatics\n"
                 "config_cross_domain seam, 11 facts (schematic 4, provenance 2 / instance 2, pragmatic 3); "
                 "the hazard is freezing a live fact",
                 fontsize=11.3, color=NAVY, y=1.04)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_design_run_split.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_design_run_split.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
