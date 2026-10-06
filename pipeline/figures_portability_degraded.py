#!/usr/bin/env python3
"""E22 Phase 3 — portability degrades with the SURFACE, and the reference repairs it in the lift.

Reads results/portability_degraded.csv. The lift is built from a progressively thinned surface
(named -> typed -> data) under three reference cases, then comprehended across the reasoning ladder;
the lifter is fixed (sol), so any fall is the surface, not a weak lifter.

The story in two panels, averaged over cases and the consumer ladder:
    left  -- meaning score falls as the surface thins WITHOUT a reference (no-ref), and lifting WITH a
             reference (lift-only, consumed bare) restores it toward the named control. Cognition builds
             the missing common ground during the lift.
    right -- confabulation rate: the hazard rises on the bare thinned surface and the reference suppresses
             it. (Abstention, an honest "cannot determine", is the virtuous alternative to confabulating;
             it is reported in the console table.)

Reference cases:  no-ref (lift without reference, consume bare) | lift-only (lift with reference, consume
bare) | lift+consume (reference in the lift AND at consumption; the ceiling).

Writes figures/fig_portability_degraded.png. Light background, no em-dashes, no "recall".
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
CSV = ROOT / "results" / "portability_degraded.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

RUNGS = ["named", "typed", "data"]                    # surface kept; glosses ALWAYS stripped
RUNG_LABEL = ["named", "typed", "data"]
SOLID_CSV = ROOT / "results" / "comprehension_portability.csv"   # Phase 1-2 solid-lift baseline
# ref_case -> (colour, marker, display label)
REFCASES = {
    "no-ref":       (RED,  "o", "no reference"),
    "lift-only":    (TEAL, "s", "reference in lift (consumed bare)"),
    "lift+consume": (NAVY, "^", "reference in lift + at consumption"),
}


def _rows():
    if not CSV.exists():
        return []
    return list(csv.DictReader(CSV.open()))


def _mean(rows, rung, ref, col):
    v = [float(r[col]) for r in rows
         if r["rung"] == rung and r["ref_case"] == ref and r.get(col) not in ("", "None", None)]
    return sum(v) / len(v) if v else float("nan")


def _solid_baseline(cases):
    """Phase 1-2 solid-lift bare meaning_score over the SAME cases (glosses intact) — the reference line the
    degraded lifts are measured against. Returns nan if the Phase 1-2 CSV is absent."""
    if not SOLID_CSV.exists():
        return float("nan")
    rows = [r for r in csv.DictReader(SOLID_CSV.open())
            if r.get("case") in cases and r.get("arm") == "bare"
            and r.get("meaning_score") not in ("", "None", None)]
    v = [float(r["meaning_score"]) for r in rows]
    return sum(v) / len(v) if v else float("nan")


def main():
    rows = _rows()
    if not rows:
        print("no portability_degraded rows yet (need results/portability_degraded.csv)")
        return
    present_refs = [rc for rc in REFCASES if any(r["ref_case"] == rc for r in rows)]
    cases = sorted({r["case"] for r in rows})
    solid = _solid_baseline(cases)
    x = np.arange(len(RUNGS))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # the solid-lift baseline (glosses intact): what a NON-degraded lift scores, the line to fall from
    if solid == solid:
        axL.axhline(solid, ls="--", color=INK, lw=1.4, zorder=2)
        axL.text(len(RUNGS) - 1.02, solid + 0.015, f"solid lift (glosses intact) = {solid:.2f}",
                 ha="right", va="bottom", fontsize=8.5, color=INK)

    for ref in present_refs:
        col, mk, lab = REFCASES[ref]
        m = [_mean(rows, rg, ref, "meaning_score") for rg in RUNGS]
        c = [_mean(rows, rg, ref, "confabulation_rate") for rg in RUNGS]
        axL.plot(x, m, "-" + mk, color=col, lw=2.2, markersize=8, label=lab, zorder=3)
        axR.plot(x, c, "-" + mk, color=col, lw=2.2, markersize=8, label=lab, zorder=3)

    axL.set_title("Stripping the meaning-bearing surface costs portability; the reference repairs it",
                  fontsize=10.5, color=NAVY)
    axL.set_ylim(0.0, 1.05)
    axL.legend(frameon=False, fontsize=9, loc="lower left")

    cmax = max([_mean(rows, rg, rc, "confabulation_rate") for rg in RUNGS for rc in present_refs] + [0.05])
    axR.set_title("Confabulation rate: the hazard on a thinned surface, tamed by the reference",
                  fontsize=10.5, color=NAVY)
    axR.set_ylim(0.0, max(0.3, cmax * 1.3))
    axR.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x)
        ax.set_xticklabels(RUNG_LABEL, fontsize=9)
        ax.set_xlabel("surface kept (glosses always stripped; lifter fixed = sol)", fontsize=9, color=FAINT)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Portability is a property of the LIFT: a meaning-poor surface breaks it, a reference in "
                 "the lift restores most of it (mean over cases and consumer ladder)",
                 fontsize=12, color=NAVY, y=1.0)
    fig.text(0.5, -0.02,
             "All three surfaces have glosses stripped, so all fall well below the solid lift; the reference "
             "repairs at every rung. The gap is not monotonic in thinning: a bare NAME invites confabulation "
             "as readily as opaque data, so 'named' is no safer than 'data'. Degradation is the surface, not "
             "a weak lifter.",
             ha="center", fontsize=9.0, color=ORANGE, style="italic")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "fig_portability_degraded.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")

    print("wrote", out.name)
    print(f"  {'rung':7}{'ref_case':14}{'meaning':10}{'abstention':12}{'confab':9}")
    for rg in RUNGS:
        for ref in present_refs:
            print(f"  {rg:7}{ref:14}{_mean(rows, rg, ref, 'meaning_score'):<10.2f}"
                  f"{_mean(rows, rg, ref, 'abstention_rate'):<12.2f}"
                  f"{_mean(rows, rg, ref, 'confabulation_rate'):<9.2f}")


if __name__ == "__main__":
    main()
