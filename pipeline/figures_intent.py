#!/usr/bin/env python3
"""Figures for the intent study. Reads the per-phase CSVs (merging any per-model segments)
and skips a figure gracefully if its data is not present yet."""
from __future__ import annotations
import csv
import glob
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fbfaf8"
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LABEL = {"gpt-5.6-sol": "sol (strong)", "gpt-5-mini": "mini", "gpt-5-nano": "nano (weak)"}
COLOR = {"gpt-5.6-sol": BLUE, "gpt-5-mini": ORANGE, "gpt-5-nano": AQUA}
SPECTRUM = ["both_cognitive", "provider_inert", "consumer_policy", "consumer_mute", "both_inert"]
SLABEL = {"both_cognitive": "both\ncognitive", "provider_inert": "provider\ninert",
          "consumer_policy": "consumer\n(policy)", "consumer_mute": "consumer\n(mute)",
          "both_inert": "both\ninert"}
POLICIES = ["exec", "resil", "bulk"]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def m(rows, k):
    xs = [fl(r[k]) for r in rows if fl(r[k]) is not None]
    return mean(xs) if xs else 0.0


def load_phase(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    return rows


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


def _clean(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --- Figure 1: negotiation decision across the spectrum (policy-recovery) ------------------
def fig_negotiation_spectrum():
    rows = [r for r in load_phase("intent_intent_hard_phase2*.csv") if r["reference"] == "none"]
    if not rows:
        print("skip fig_intent_negotiation: no phase-2 data"); return
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for md in MODELS:
        ys = [m([r for r in rows if r["model"] == md and r["placement"] == pl], "decision_accuracy")
              for pl in SPECTRUM]
        # sol tracks mini exactly here; nudge it up a hair (and thin it) so the blue line stays visible
        # instead of hiding under the orange one
        dy = 0.018 if md == "gpt-5.6-sol" else 0.0
        lw = 1.6 if md == "gpt-5.6-sol" else 2.0
        ax.plot(range(len(SPECTRUM)), [y + dy for y in ys], "-o", color=COLOR[md], label=LABEL[md],
                markersize=6.5 if md == "gpt-5.6-sol" else 7, linewidth=lw,
                zorder=5 if md == "gpt-5.6-sol" else 3)
    ax.set_xticks(range(len(SPECTRUM))); ax.set_xticklabels([SLABEL[p] for p in SPECTRUM], fontsize=9)
    ax.set_ylabel("decision closed correctly (accept/reject)")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("The negotiation across the cognition spectrum:\n"
                 "a pre-placed movable policy closes decisions a mute consumer must refer", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    # neutral ink (not AQUA — AQUA is the nano series); text sits in the empty upper-right and the
    # arrow runs above the plunging lines, targeting the consumer-(policy) column it explains
    INK = "#14314f"
    ax.annotate("pre-placed policy recovers\nthe decision a mute consumer loses",
                xy=(2.06, 0.86), xytext=(2.62, 0.94),
                fontsize=8.5, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    _clean(ax)
    save(fig, "fig_intent_negotiation.png")


# --- Figure 2: pragmatic sensitivity (decision tracks the policy) --------------------------
def fig_pragmatic():
    rows = [r for r in load_phase("intent_intent_hard_phase2*.csv")
            if r["placement"] == "both_cognitive" and r["reference"] == "none"]
    if not rows:
        print("skip fig_intent_pragmatic: no phase-2 data"); return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = range(len(POLICIES))
    w = 0.25
    for j, md in enumerate(MODELS):
        ys = [m([r for r in rows if r["model"] == md and r["policy"] == pol], "decision_accuracy")
              for pol in POLICIES]
        ax.bar([i + (j - 1) * w for i in x], ys, width=w, color=COLOR[md], label=LABEL[md])
    ax.set_xticks(list(x)); ax.set_xticklabels(["exec\n(latency-first)", "resil\n(resilience-first)",
                                                "bulk\n(cost-sensitive)"], fontsize=9)
    ax.set_ylabel("decision accuracy under the policy")
    ax.set_ylim(0, 1.08)
    ax.set_title("The decision tracks the movable policy:\n"
                 "the same infeasible intents, judged correctly under each policy", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _clean(ax)
    save(fig, "fig_intent_pragmatic.png")


# --- Figure 2: the T1 lifecycle set-piece arc (illustration, not a data chart) -------------
def fig_lifecycle():
    fig, ax = plt.subplots(figsize=(11.6, 3.4))
    hops = [("h0", "buy", "met", "hold  ·  r1a"),
            ("h1", "provider:\nlatency degrades", "breach", "reroute\nr1a → r1c"),
            ("h2", "consumer:\nsix-nines demand", "breach", "refer\n(needs the consumer)"),
            ("h3", "provider:\nfault clears", "met", "restore\nr1c → r1a")]
    fcol = {"met": AQUA, "breach": ORANGE}
    for i, (hid, ev, ful, dec) in enumerate(hops):
        ax.scatter([i], [1], s=560, color=fcol[ful], zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(i, 1, hid, ha="center", va="center", fontsize=9, color="white", zorder=4, weight="bold")
        ax.text(i, 1.34, ev, ha="center", va="bottom", fontsize=8.5, color="#333")
        ax.text(i, 0.66, dec, ha="center", va="top", fontsize=8.5, color="#14314f")
        if i:
            ax.annotate("", xy=(i - 0.14, 1), xytext=(i - 0.86, 1),
                        arrowprops=dict(arrowstyle="->", color="#999", lw=1.4))
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0.2, 1.8)
    ax.axis("off")
    ax.set_title("The T1 set-piece — an order-execution service reconciling itself across its life "
                 "(green = met, orange = breach; the label is the reconciliation each hop sets off)",
                 fontsize=10.5)
    save(fig, "fig_intent_lifecycle.png")


# --- Figure 4: refine-down satisfaction + experiment-only across the spectrum --------------
def fig_refine():
    rows = [r for r in load_phase("intent_intent_hard_phase1*.csv") if r["reference"] == "none"]
    if not rows:
        print("skip fig_intent_refine: no phase-1 data"); return
    placements = ["both_cognitive", "provider_inert", "both_inert"]
    plabel = {"both_cognitive": "both\ncognitive", "provider_inert": "provider\ninert",
              "both_inert": "both\ninert"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    # left: overall satisfaction accuracy
    for md in MODELS:
        y1 = [m([r for r in rows if r["model"] == md and r["placement"] == pl], "satisfaction_accuracy")
              for pl in placements]
        ax1.plot(range(3), y1, "-o", color=COLOR[md], label=LABEL[md])
    # right: experiment-only correctness as a fraction of eo_total
    for md in MODELS:
        ys = []
        for pl in placements:
            sel = [r for r in rows if r["model"] == md and r["placement"] == pl]
            tot = sum(int(fl(r["eo_total"]) or 0) for r in sel)
            cor = sum(int(fl(r["eo_correct"]) or 0) for r in sel)
            ys.append(cor / tot if tot else 0.0)
        ax2.plot(range(3), ys, "-o", color=COLOR[md], label=LABEL[md])
    for ax, ylab, ttl in ((ax1, "satisfaction accuracy", "Refine-down: satisfaction correct"),
                          (ax2, "experiment-only correct (fraction)",
                           "Experiment-only intents: need the live probe")):
        ax.set_xticks(range(3)); ax.set_xticklabels([plabel[p] for p in placements], fontsize=9)
        ax.set_ylabel(ylab); ax.set_ylim(-0.03, 1.08); _clean(ax)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Refine-down across the spectrum: advertised evidence suffices for the easy intents; "
                 "the experiment-only ones need a live feasibility check", fontsize=11, y=1.02)
    save(fig, "fig_intent_refine.png")


if __name__ == "__main__":
    # Only the two figures that show a shape worth seeing: the negotiation policy-recovery
    # step across the spectrum, and the lifecycle set-piece arc. Everything else the study
    # measured is a number, and lives in the report's text and tables.
    fig_negotiation_spectrum()
    fig_lifecycle()
