#!/usr/bin/env python3
"""Breadth-replication figure: the signature result of each of the four settings, reproduced
on the two new (independent) cases built for that setting. Values are the real agents run on
the new cases, scored against the validated answer key, at the fully-cognitive end.

Writes figures/fig_breadth.png.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent.parent / "figures"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b8c2cc"
NAVY, SURFACE, INK = "#14314f", "#fbfaf8", "#1a1a1a"


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    (a1, a3), (a2, a4) = axes

    # --- Setting 1: the strong agent reconciles the new pairs; the reference completes the close -
    cases = ["config_l3vpn\n(IP/MPLS VPN)", "config_evpn\n(carrier Ethernet)"]
    sol1_noref = [0.77, 0.31]; sol1_ref = [0.91, 1.00]
    x = np.arange(2); w = 0.36
    a1.bar(x - w/2, sol1_noref, w, color=GREY, label="no shared reference")
    a1.bar(x + w/2, sol1_ref, w, color=BLUE, label="with the reference")
    a1.axhline(1.0, color=NAVY, ls=":", lw=1)
    a1.set_title("Scenario 1 - configuration\n"
                 "two agents reconcile the new standard-model pairs at perfect precision; where they\n"
                 "defer without shared ground, the reference completes the close (strong agent shown)",
                 fontsize=9.5)
    a1.set_ylabel("resolved fraction, strong agent\n(share of true matches found)", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels(cases, fontsize=8.5)
    a1.set_ylim(0, 1.08); a1.legend(frameon=True, framealpha=0.95, edgecolor="none", facecolor=SURFACE, fontsize=8.5, loc="center")

    # --- Setting 3: the mirror - strong under-commits, reference completes ---------------------
    cases3 = ["config_xdom_ran\n(RAN <-> core)", "config_xdom_dc\n(fabric <-> overlay)"]
    sol_noref = [0.40, 0.20]; sol_ref = [1.00, 1.00]
    a3.bar(x - w/2, sol_noref, w, color=GREY, label="no shared reference")
    a3.bar(x + w/2, sol_ref, w, color=BLUE, label="with the reference")
    a3.axhline(1.0, color=NAVY, ls=":", lw=1)
    a3.set_title("Scenario 3 - standard-free (no standard between the two sides)\n"
                 "without a shared reference the STRONG agent under-commits (refuses to guess the seam):\n"
                 "perfect precision but low resolved fraction; the constructed reference completes the close",
                 fontsize=9.5)
    a3.set_ylabel("resolved fraction, strong agent\n(share of true matches found)", fontsize=9)
    a3.set_xticks(x); a3.set_xticklabels(cases3, fontsize=8.5)
    a3.set_ylim(0, 1.08); a3.legend(frameon=True, framealpha=0.95, edgecolor="none", facecolor=SURFACE, fontsize=8.5, loc="center")

    # --- Setting 2: refinement completes for capable agents, weak lags on live-check ----------
    cases2 = ["intent_metro", "intent_dci"]
    sol = [1.00, 1.00]; mini = [1.00, 1.00]; nano = [0.83, 0.83]
    xx = np.arange(2); w3 = 0.26
    a2.bar(xx - w3, sol, w3, color=BLUE, label="sol (strong)")
    a2.bar(xx, mini, w3, color=ORANGE, label="mini (mid)")
    a2.bar(xx + w3, nano, w3, color=AQUA, label="nano (weak)")
    a2.axhline(1.0, color=NAVY, ls=":", lw=1)
    a2.set_title("Scenario 2 - intent (customer wishes vs operator catalogue)\n"
                 "working out which options meet the wish: strong and mid are perfect and get every\n"
                 "'only-a-live-check-settles-it' case; the weak agent lags (0.83; 2 of 4 live-check cases)",
                 fontsize=9.5)
    a2.set_ylabel("refinement accuracy\n(right satisfy/not-satisfy verdicts)", fontsize=9)
    a2.set_xticks(xx); a2.set_xticklabels(cases2, fontsize=9)
    a2.set_ylim(0.6, 1.04); a2.legend(frameon=True, framealpha=0.9, edgecolor="none", facecolor=SURFACE, fontsize=8.5, loc="lower left", ncol=3)

    # --- Setting 4: alarm != anomaly look-alike avoided; decomposition is the residual --------
    cases4 = ["obs_routing", "obs_compute"]
    prec = [1.00, 1.00]; rec = [0.75, 0.75]
    a4.bar(xx - w/2, prec, w, color=BLUE, label="precision (look-alike avoided)")
    a4.bar(xx + w/2, rec, w, color=GREY, label="resolved fraction (decomposition half-done)")
    a4.axhline(1.0, color=NAVY, ls=":", lw=1)
    a4.set_title("Scenario 4 - observability (an alarm is not an anomaly)\n"
                 "across every agent and both fault domains the deep look-alike is avoided (precision 1.0,\n"
                 "zero traps taken); the one-to-many decomposition is the cognition-demanding residual",
                 fontsize=9.5)
    a4.set_ylabel("precision / resolved fraction (all agents)", fontsize=9)
    a4.set_xticks(xx); a4.set_xticklabels(cases4, fontsize=9)
    a4.set_ylim(0, 1.08); a4.legend(frameon=True, framealpha=0.9, edgecolor="none", facecolor=SURFACE, fontsize=8.5, loc="lower center")

    for ax in (a1, a2, a3, a4):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Breadth: each scenario's signature result, reproduced on two new independently-built cases",
                 fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG / "fig_breadth.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / "fig_breadth.png").name)


if __name__ == "__main__":
    main()
