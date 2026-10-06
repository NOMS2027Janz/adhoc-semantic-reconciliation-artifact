#!/usr/bin/env python3
"""E17 -- cold-start lift from a thin schema (report figure).

Reads results/cold_start_lift.csv and draws, one panel per model, a stacked composition of the
lift outcome across the thinning ladder (named -> typed -> data -> topology): what share of
concepts the agent RECOVERS (asserts a faithful gloss), DEFERS (abstains honestly), or
HALLUCINATES (asserts an unsupported one). The three sum to one.

The story the figure carries: on the full surface every tier recovers. As the surface thins the
capable model converts recovery into DEFERRAL -- it abstains where the meaning is gone -- while the
weak model keeps asserting, so its bar fills with HALLUCINATION. Capability at lift time is not more
recovery; it is knowing when you cannot lift. Light background.

    python figures_cold_start_lift.py                     # default results CSV
    python figures_cold_start_lift.py results/e17_x.csv   # explicit CSV (testing)

Writes figures/fig_cold_start_lift.png.
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
CSV = ROOT / "results" / "cold_start_lift.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
TITLE = {"gpt-5.6-sol": "sol (strong)", "gpt-5-mini": "mini (mid)", "gpt-5-nano": "nano (weak)"}
RUNGS = ["named", "typed", "data", "topology"]
RUNGLAB = ["named", "typed", "data", "topol."]
SEGS = [("recover_rate", "recovers", TEAL),
        ("defer_rate", "defers", NAVY),
        ("hallucinate_rate", "hallucinates", RED)]


def _means(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["rung"])].append(r)

    def m(md, rung, k):
        v = [float(x[k]) for x in agg.get((md, rung), []) if x.get(k) not in ("", "None", None)]
        return sum(v) / len(v) if v else float("nan")
    return {md: {k: [m(md, rg, k) for rg in RUNGS] for k, _, _ in SEGS} for md in ORDER}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


CASE_DESC = {"config_cross_domain": "config_cross_domain (Meridian/Cascade)",
             "config_rest": "config_rest (MEF Sonata / TMF641, REST)"}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    import csv as _csv
    _rows = list(_csv.DictReader(open(csv_path)))
    case = _rows[0]["case"] if _rows else "config_cross_domain"
    ncon = _rows[0].get("n_concepts", "") if _rows else ""
    desc = CASE_DESC.get(case, case)
    suffix = "" if case == "config_cross_domain" else f"_{case.replace('config_', '')}"
    data = _means(csv_path)
    present = [md for md in ORDER
              if not all(np.isnan(v) for v in data[md]["recover_rate"])]
    if not present:
        raise SystemExit(f"no usable rows in {csv_path}")

    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 5.2), sharey=True)
    if len(present) == 1:
        axes = [axes]
    x = np.arange(len(RUNGS))
    for ax, md in zip(axes, present):
        bottom = np.zeros(len(RUNGS))
        for key, lab, col in SEGS:
            vals = np.array([0.0 if np.isnan(v) else v for v in data[md][key]])
            ax.bar(x, vals, bottom=bottom, width=0.64, color=col, label=lab, zorder=3,
                   edgecolor=SURFACE, linewidth=0.6)
            for xi, v, b in zip(x, vals, bottom):
                if v >= 0.08:
                    ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            bottom += vals
        ax.set_title(TITLE[md], fontsize=11.5, color=NAVY)
        ax.set_xticks(x)
        ax.set_xticklabels(RUNGLAB, fontsize=9.5, color=INK)
        ax.set_ylim(0, 1.01)
        _style(ax)
    axes[0].set_ylabel("share of concepts", fontsize=10, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Cold-start lift from a thinning schema: recovery vs honest deferral vs confident "
                 "invention, as the surface is stripped\n"
                 f"{desc}, both sides ({ncon} concepts); judge-graded; stripped left to right: "
                 "names, types, data",
                 fontsize=11.8, color=NAVY, y=1.03)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    outp = FIG / f"fig_cold_start_lift{suffix}.png"
    fig.savefig(outp, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", outp.relative_to(ROOT))


if __name__ == "__main__":
    main()
