#!/usr/bin/env python3
"""E18 -- the minimum viable overlay (report figure).

Reads results/minimum_overlay.csv and plots, per model, how reconciliation over a THIN source holds
up as the governed overlay is thinned on two axes:

  TOP ROW     COVERAGE -- x = number of overlay entries kept (0..5), dropped easy-first so the two
              non-obvious seams are the last to lose their entry. LEFT recall, RIGHT seam recall
              (the two hard seams). The knee shows how few entries rescue the thin source, and that
              the load-bearing entries are the ambiguous seams.
  BOTTOM ROW  RICHNESS at full coverage -- x = per-entry field set from id-only up to full. LEFT
              recall, RIGHT seam recall. Shows how little each entry must carry.

Line per model. Light background.

    python figures_minimum_overlay.py                     # default results CSV
    python figures_minimum_overlay.py results/e18_x.csv   # explicit CSV (testing)

Writes figures/fig_minimum_overlay.png.
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
CSV = ROOT / "results" / "minimum_overlay.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = ["sol (strong)", "mini (mid)", "nano (weak)"]
COLOR = [NAVY, TEAL, ORANGE]
MARK = ["-o", "-s", "-^"]
COV_X = [0, 1, 2, 3, 4, 5]
RICH_ORDER = ["id_only", "label", "definition", "def_example", "full"]
RICH_LAB = ["id-only", "+label", "+defn", "+defn\n+ex", "full"]


def _rows(csv_path):
    return list(csv.DictReader(open(csv_path)))


def _series(rows, model, sweep, xkeys, key, xfield):
    sel = [r for r in rows if r["model"] == model and r["sweep"] == sweep]
    agg = defaultdict(list)
    for r in sel:
        agg[str(r[xfield])].append(r)
    out = []
    for xk in xkeys:
        vs = [float(r[key]) for r in agg.get(str(xk), []) if r.get(key) not in ("", "None", None)]
        out.append(sum(vs) / len(vs) if vs else np.nan)
    return out


def _style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    rows = _rows(csv_path)
    present = [m for m in ORDER if any(r["model"] == m for r in rows)]
    if not present:
        raise SystemExit(f"no usable rows in {csv_path}")

    fig, ((a00, a01), (a10, a11)) = plt.subplots(2, 2, figsize=(12.6, 8.6))

    for m, lab, c, mk in zip(ORDER, LABEL, COLOR, MARK):
        if m not in present:
            continue
        idx = ORDER.index(m)
        # coverage row: x = n_entries via 'level' (0..5)
        a00.plot(COV_X, _series(rows, m, "coverage", COV_X, "recall", "level"),
                 mk, color=c, lw=2, markersize=6.5, label=lab, zorder=3)
        a01.plot(COV_X, _series(rows, m, "coverage", COV_X, "seam_recall", "level"),
                 mk, color=c, lw=2, markersize=6.5, label=lab, zorder=3)
        # richness row: id_only..def_example from sweep 'fields'; 'full' from coverage level 5
        def rich(key):
            fields = _series(rows, m, "fields", RICH_ORDER[:-1], key, "level")
            full = _series(rows, m, "coverage", ["5"], key, "level")[0]
            return fields + [full]
        xr = list(range(len(RICH_ORDER)))
        a10.plot(xr, rich("recall"), mk, color=c, lw=2, markersize=6.5, label=lab, zorder=3)
        a11.plot(xr, rich("seam_recall"), mk, color=c, lw=2, markersize=6.5, label=lab, zorder=3)

    a00.set_title("Coverage: recall vs overlay entries kept", fontsize=11, color=NAVY)
    a01.set_title("Coverage: hard-seam recall vs entries kept", fontsize=11, color=NAVY)
    a10.set_title("Richness: recall vs per-entry fields", fontsize=11, color=NAVY)
    a11.set_title("Richness: hard-seam recall vs fields", fontsize=11, color=NAVY)
    for ax in (a00, a01):
        _style(ax, "overlay entries kept  (dropped easy-first; hard seams last)")
        ax.set_xticks(COV_X)
    for ax in (a10, a11):
        _style(ax, "per-entry field set (at full coverage)")
        ax.set_xticks(list(range(len(RICH_ORDER))))
        ax.set_xticklabels(RICH_LAB, fontsize=8.5)
    a00.set_ylabel("recall", fontsize=10, color=INK)
    a10.set_ylabel("recall", fontsize=10, color=INK)
    a01.set_ylabel("hard-seam recall", fontsize=10, color=INK)
    a11.set_ylabel("hard-seam recall", fontsize=10, color=INK)
    a00.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("The minimum viable overlay over a thin source: how few entries, and how little each,"
                 " rescue reconciliation\nconfig_cross_domain, inert source side, one_inert; opaque "
                 "reference ids; three trials per cell",
                 fontsize=12.0, color=NAVY, y=1.02)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_minimum_overlay.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_minimum_overlay.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
