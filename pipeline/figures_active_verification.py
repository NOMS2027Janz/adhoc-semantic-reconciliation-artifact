#!/usr/bin/env python3
"""E21 -- active verification between two cognitive systems (report figure).

Reads results/active_verification.csv and plots, per hard case and across the capability ladder
(strong -> mid -> weak), the two quantities the note turns on:

TOP ROW    resolved fraction -- the share of the seam correspondences the peers close WITHOUT an
           authored reference. A line flat at 1.0 across the ladder = a capability-INDEPENDENT route
           (matches an authored reference); a line that sags toward the weak end = capability-GATED.
BOTTOM ROW surviving false cognates -- the guard: how many planted cognates slip through. Lower is
           better; the authored reference holds this at zero.

Lines, per case:
  * two-agent            -- negotiation only (exchange meaning, no decisive test). The L1 baseline.
  * two-agent+exp        -- + the decisive virtual experiment (active verification). THE arm.
  * two-agent+frame      -- peers construct their own shared reference, then negotiate (synthesis).
  * two-agent+frame+exp  -- construct AND verify (full synthesis).
  * two-agent/ref        -- reference handed over (ceiling, dashed).
Arms absent from the CSV (e.g. the +frame synthesis arms before --with-frame is run) are simply
skipped, so the figure renders at every stage of the batch.

    python figures_active_verification.py                              # default results CSV
    python figures_active_verification.py results/active_verification.csv

Writes figures/fig_active_verification.png. Light background.
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
CSV = ROOT / "results" / "active_verification.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
TIERLAB = ["sol\n(strong)", "mini\n(mid)", "nano\n(weak)"]
CASES = ["config_cross_domain", "config_big_hard", "config_rest"]
CASELAB = {"config_cross_domain": "cross-domain seam",
           "config_big_hard": "configuration (hard)",
           "config_rest": "REST (identical-enum cognate)"}

# (stack label, colour, linestyle, marker, legend label) -- drawn in this order where present.
# NOTE: single/ref (reference handed to a single agent) is the TRUE ceiling -- flat ~1.0 at every tier.
# two-agent/ref is the reference used INSIDE a negotiation, which sags for weaker tiers (a finding, not a
# ceiling), so it is drawn as a distinct dotted line, not as "the ceiling".
ARMS = [
    ("two-agent",           FAINT,  "-",  "o", "negotiation only"),
    ("two-agent+exp",       NAVY,   "-",  "o", "+ decisive experiment"),
    ("two-agent+frame",     ORANGE, "-",  "s", "+ constructed frame"),
    ("two-agent+frame+exp", TEAL,   "-",  "^", "+ frame + experiment"),
    ("two-agent/ref",       RED,    ":",  "x", "reference (in negotiation)"),
    ("single/ref",          INK,    "--", "D", "reference — single agent (ceiling)"),
]


def _load(csv_path: Path):
    agg = defaultdict(list)  # (case, stack, model) -> [rows]
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            agg[(r["case"], r["stack"], r["model"])].append(r)
    return agg


def _mean(agg, case, stack, model, key):
    vals = [float(x[key]) for x in agg.get((case, stack, model), [])
            if x.get(key) not in ("", "None", None)]
    return sum(vals) / len(vals) if vals else float("nan")


def _series(agg, case, stack, key):
    return [_mean(agg, case, stack, md, key) for md in ORDER]


def _style(ax, ymax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(TIERLAB, fontsize=9)
    ax.set_xlim(-0.3, len(ORDER) - 0.7)
    ax.set_ylim(-0.04 * ymax, ymax * 1.08)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not csv_path.exists():
        print(f"no results yet at {csv_path} -- run pipeline/active_verification.py first",
              file=sys.stderr)
        return 1
    agg = _load(csv_path)
    present = {stack for (_, stack, _) in agg}
    arms = [a for a in ARMS if a[0] in present]

    # max surviving-cognate count across the data, for the bottom-row y-scale (min 1)
    sfc_max = 1.0
    for case in CASES:
        for a in arms:
            for v in _series(agg, case, a[0], "surviving_false_cognates"):
                if v == v:  # not nan
                    sfc_max = max(sfc_max, v)

    fig, axes = plt.subplots(2, len(CASES), figsize=(4.4 * len(CASES), 7.2),
                             facecolor=SURFACE, squeeze=False)
    x = np.arange(len(ORDER))
    for j, case in enumerate(CASES):
        top, bot = axes[0][j], axes[1][j]
        for stack, colour, ls, mk, _lab in arms:
            rf = _series(agg, case, stack, "resolved_fraction")
            sfc = _series(agg, case, stack, "surviving_false_cognates")
            top.plot(x, rf, ls, color=colour, marker=mk, lw=2, ms=6, alpha=0.95)
            bot.plot(x, sfc, ls, color=colour, marker=mk, lw=2, ms=6, alpha=0.95)
        top.axhline(1.0, color=FAINT, lw=0.8, ls=":", alpha=0.7)
        _style(top, 1.0)
        _style(bot, sfc_max)
        top.set_title(CASELAB.get(case, case), fontsize=11, color=INK, pad=8)
        if j == 0:
            top.set_ylabel("resolved fraction", fontsize=10, color=INK)
            bot.set_ylabel("surviving false cognates", fontsize=10, color=INK)

    # one shared legend under the figure
    handles = [plt.Line2D([0], [0], color=c, ls=ls, marker=mk, lw=2, ms=6, label=lab)
               for (_s, c, ls, mk, lab) in arms]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Active verification between two cognitive systems, across the capability ladder",
                 fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    FIG.mkdir(exist_ok=True)
    outp = FIG / "fig_active_verification.png"
    fig.savefig(outp, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", outp.relative_to(ROOT), f"({len(arms)} arms present:",
          ", ".join(a[0] for a in arms) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
