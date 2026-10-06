#!/usr/bin/env python3
"""E16 -- inert vs cognitive source, crossed with a governed overlay (report figure).

Reads results/inert_vs_cognitive.csv and plots, per model across the three conditions
(inert / cognitive / rich) and the two reference modes (overlay off / on), the two things
that decide whether a thin source can be reconciled:

TOP ROW    recall -- the share of the five seam correspondences resolved. Exposes the strong
           model's honest deferral: with no overlay it proposes almost nothing from a thin
           source (it will not commit a cross-domain binding it cannot justify), and the
           overlay is exactly what lets it commit.
BOTTOM ROW taking the planted false cognate (grade<->grade) -- the error the meaning is there
           to prevent.
LEFT COLUMN  reference OFF -- meaning comes only from the source (thin / elicited / rich).
RIGHT COLUMN reference ON -- the ad-hoc reference is present as a governed overlay.

The finding the figure carries: the governed overlay is the robust, capability-independent
route -- it lifts the strong model out of deferral (recall 0.2 -> 1.0 from an inert source)
and blocks the false cognate across the ladder. Live source-cognition (the inert->cognitive
shift) is the substitute where there is no overlay, but it only bites in the capability band
that can use elicited meaning: it rescues the mid model's false cognate, barely moves the
deferring strong model, and does not save the weak model on its own. Light background.

    python figures_inert_cognitive.py                     # default results CSV
    python figures_inert_cognitive.py results/e16_x.csv   # explicit CSV (testing)

Writes figures/fig_inert_cognitive.png.
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
CSV = ROOT / "results" / "inert_vs_cognitive.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = ["sol (strong)", "mini (mid)", "nano (weak)"]
COLOR = [NAVY, TEAL, ORANGE]
CONDS = ["inert", "cognitive", "rich"]
CONDLAB = ["inert", "cognitive", "rich"]


def _means(csv_path: Path, ref: str, key: str):
    want = "True" if ref == "on" else "False"
    rows = [r for r in csv.DictReader(open(csv_path))
            if str(r.get("uses_reference", "False")) == want]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["condition"])].append(r)

    def m(md, cond):
        v = [float(x[key]) for x in agg.get((md, cond), []) if x.get(key) not in ("", "None", None)]
        return sum(v) / len(v) if v else float("nan")
    return {md: [m(md, c) for c in CONDS] for md in ORDER}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def _grouped(ax, data, present):
    n = len(present)
    w = 0.8 / n
    xc = np.arange(len(CONDS))
    for i, md in enumerate(present):
        idx = ORDER.index(md)
        vals = [0.0 if np.isnan(v) else v for v in data[md]]
        xs = xc - 0.4 + w / 2 + i * w
        ax.bar(xs, vals, width=w * 0.92, color=COLOR[idx], label=LABEL[idx], zorder=3)
        for x, v in zip(xs, data[md]):
            if not np.isnan(v):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=INK)
    ax.set_xticks(xc)
    ax.set_xticklabels(CONDLAB, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


CASE_DESC = {
    "config_cross_domain": "config_cross_domain  (Meridian transport thin side vs Cascade IP)",
    "config_rest": "config_rest  (MEF Sonata product order thin side vs TMF641 service order)",
}
CASE_COGNATE = {"config_cross_domain": "grade<->grade", "config_rest": "state<->state"}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    import csv as _csv
    _rows = list(_csv.DictReader(open(csv_path)))
    case = _rows[0]["case"] if _rows else "config_cross_domain"
    desc = CASE_DESC.get(case, case)
    cognate = CASE_COGNATE.get(case, "planted")
    suffix = "" if case == "config_cross_domain" else f"_{case.replace('config_', '')}"

    data = {(row, ref): _means(csv_path, ref, key)
            for row, key in (("recall", "recall"), ("grade", "took_grade_cognate"))
            for ref in ("off", "on")}
    present = [md for md in ORDER
              if not all(np.isnan(v) for v in data[("recall", "off")][md])]
    if not present:
        raise SystemExit(f"no usable rows in {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharey="row")
    (a00, a01), (a10, a11) = axes

    _grouped(a00, data[("recall", "off")], present)
    _grouped(a01, data[("recall", "on")], present)
    _grouped(a10, data[("grade", "off")], present)
    _grouped(a11, data[("grade", "on")], present)

    a00.set_title("no overlay  (reference off)", fontsize=11.5, color=NAVY)
    a01.set_title("governed overlay  (reference on)", fontsize=11.5, color=NAVY)
    a00.set_ylabel("resolved fraction\n(share of 5 correspondences resolved)", fontsize=10, color=INK)
    a10.set_ylabel(f"false cognate taken\n({cognate}, share of trials)", fontsize=10, color=INK)
    a11.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=1)

    for ax in (a00, a01, a10, a11):
        _style(ax)

    fig.suptitle("Producing the inputs for a thin source: the overlay rescues across the ladder; "
                 "live source-cognition only in the band that can use it\n"
                 f"{desc}, four trials per cell",
                 fontsize=12.2, color=NAVY, y=1.02)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    outp = FIG / f"fig_inert_cognitive{suffix}.png"
    fig.savefig(outp, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
