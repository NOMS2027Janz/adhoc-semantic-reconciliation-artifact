#!/usr/bin/env python3
"""A deliverability-reasoning provider (report 2/4 Figure, master reference).

Reads results/deliverability_negotiation.csv and shows, per model:
LEFT  divergence from the catalogue oracle by scenario -- near zero in the `clear` regime (the E13
      result reproduced: nothing to reason about) and rising sharply once a path saturates, because the
      provider must counter-propose or decline. This is the second agent starting to matter.
RIGHT how often an UNDELIVERABLE service is offered -- 0.52 for the catalogue oracle (which ignores
      capacity), zero for every reasoning provider.

The finding: E13's negotiation null was specific to a fixed-catalogue provider. Where the provider must
weigh its own deliverability, a reasoning agent diverges substantially from the catalogue oracle -- the
negotiation is genuinely two-sided there -- and even the weak model avoids undeliverable offers. Light
background, no em-dashes.

Writes figures/fig_deliverability.png.
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
CSV = ROOT / "results" / "deliverability_negotiation.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = ["sol (strong)", "mini (mid)", "nano (weak)"]
COLOR = [NAVY, TEAL, ORANGE]
SCEN = ["clear", "constrained", "outage"]
SCEN_LABEL = ["clear", "constrained", "outage"]


def _load():
    rows = list(csv.DictReader(open(CSV)))
    ag = [r for r in rows if r["arm"] == "agent"]

    def m(rs, k):
        xs = [float(r[k]) for r in rs if r.get(k) not in ("", "None", None)]
        return sum(xs) / len(xs) if xs else 0.0

    div = {md: [m([r for r in ag if r["model"] == md and r["scenario"] == s],
                  "diverges_from_catalogue") for s in SCEN] for md in ORDER}
    # undeliverable-offer rate = 1 - deliverable, overall, per model; and the catalogue oracle's
    undeliv_agent = {md: 1 - m([r for r in ag if r["model"] == md], "offer_deliverable") for md in ORDER}
    cat = [r for r in rows if r["arm"] == "catalogue_oracle"]
    undeliv_cat = m(cat, "catalogue_undeliverable")
    return div, undeliv_agent, undeliv_cat


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)


def main():
    div, undeliv_agent, undeliv_cat = _load()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})

    # LEFT: grouped bars, scenario x model, divergence from catalogue oracle
    x = np.arange(len(SCEN))
    w = 0.26
    for i, (md, lab, c) in enumerate(zip(ORDER, LABEL, COLOR)):
        xs = x + (i - 1) * w
        ys = div[md]
        axL.bar(xs, ys, width=w, color=c, label=lab, edgecolor="white", zorder=3)
        for xi, yi in zip(xs, ys):
            axL.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    axL.set_title("The second agent matters once deliverability binds",
                  fontsize=11, color=NAVY)
    axL.set_ylabel("divergence from the catalogue oracle", fontsize=10, color=INK)
    axL.set_ylim(0, 1.12)
    axL.set_xticks(x); axL.set_xticklabels(SCEN_LABEL, fontsize=9.5)
    axL.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style(axL)

    # RIGHT: undeliverable-offer rate, catalogue oracle vs the reasoning providers
    bars = ["catalogue\noracle"] + LABEL
    vals = [undeliv_cat] + [undeliv_agent[md] for md in ORDER]
    cols = [RED, NAVY, TEAL, ORANGE]
    xb = np.arange(len(bars))
    axR.bar(xb, vals, width=0.6, color=cols, edgecolor="white", zorder=3)
    for xi, yi in zip(xb, vals):
        axR.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    axR.set_title("Reasoning about capacity avoids undeliverable offers",
                  fontsize=11, color=NAVY)
    axR.set_ylabel("share of offers that are undeliverable", fontsize=10, color=INK)
    axR.set_ylim(0, 0.62)
    axR.set_xticks(xb); axR.set_xticklabels(bars, fontsize=9)
    _style(axR)
    axR.annotate("the naive provider offers a service it\ncannot deliver in over half of cells;\nevery reasoning provider offers none",
                 xy=(0.62, 0.52), xycoords="axes fraction", ha="center", va="center", fontsize=8.5,
                 color=FAINT, style="italic")

    fig.suptitle("A deliverability-reasoning provider vs the catalogue oracle  (intent negotiation)",
                 fontsize=12.5, color=NAVY, y=1.02)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_deliverability.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_deliverability.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
