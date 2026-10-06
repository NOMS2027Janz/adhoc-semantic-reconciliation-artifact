#!/usr/bin/env python3
"""Figures for the observability study. Scene schematic is data-independent; the two results
figures read the per-model phase CSVs."""
from __future__ import annotations
import csv, glob
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, SURFACE = "#9a958d", "#fbfaf8"
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
LAB = {"gpt-5.6-sol": "sol\n(strong)", "gpt-5-mini": "mini", "gpt-5-nano": "nano\n(weak)"}
COL = {"gpt-5.6-sol": BLUE, "gpt-5-mini": ORANGE, "gpt-5-nano": AQUA}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(pat):
    rows = []
    for f in glob.glob(str(RES / pat)):
        rows += list(csv.DictReader(open(f)))
    return rows


def loadm(M, phase):
    f = RES / f"obs_config_observability_phase{phase}_{M}.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []


def m(rs, k):
    xs = [fl(r[k]) for r in rs if fl(r[k]) is not None]
    return mean(xs) if xs else 0.0


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", (FIG / name).name)


def _clean(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --- Figure 1: the scene — the overloaded legacy alarm decomposing into the NMOP ladder -------
def fig_scene():
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off"); ax.set_facecolor(SURFACE)
    # legacy side (title sits clear above the box; text has margins inside it)
    ax.add_patch(FancyBboxPatch((0.4, 3.2), 3.0, 2.7, boxstyle="round,pad=0.1",
                                fc="#fdf1ea", ec=ORANGE, lw=1.6))
    ax.text(1.9, 6.4, "Agent F — legacy", ha="center", fontsize=11, weight="bold", color="#14314f")
    ax.text(1.9, 5.55, "one ALARM, bundling:", ha="center", fontsize=9.0, color="#333", style="italic")
    for j, t in enumerate(["event", "state", "fixed severity", "probable-cause"]):
        ax.text(1.9, 5.02 - j * 0.42, "· " + t, ha="center", fontsize=8.6, color="#555")
    # arrow
    ax.annotate("", xy=(4.9, 4.7), xytext=(3.5, 4.7),
                arrowprops=dict(arrowstyle="-|>", color="#777", lw=2))
    ax.text(4.2, 5.05, "decompose\n(lossy up)", ha="center", fontsize=8, color="#777")
    # NMOP ladder
    ladder = ["event", "anomaly", "symptom", "fault", "alarm (State)", "problem", "cause", "incident"]
    x0 = 5.2
    for j, t in enumerate(ladder):
        yy = 8.0 - j * 0.92
        is_trap = (t == "anomaly")
        is_target = (t == "alarm (State)")
        ec = ORANGE if is_trap else (AQUA if is_target else BLUE)
        fc = "#fdf1ea" if is_trap else ("#eafbf3" if is_target else "white")
        ax.add_patch(FancyBboxPatch((x0, yy - 0.3), 3.4, 0.6, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.3))
        ax.text(x0 + 0.2, yy, t, ha="left", va="center", fontsize=9.2, color="#14314f")
    # labels beside their own boxes: green next to alarm-State (row 4), orange next to anomaly (row 1)
    ax.text(8.75, 8.0 - 1 * 0.92, "TRAP: a deviation, not a state —\nan alarm is not an anomaly",
            fontsize=8.0, color=ORANGE, va="center")
    ax.text(8.75, 8.0 - 4 * 0.92, "the correct core\nof an alarm", fontsize=8.0, color=AQUA, va="center")
    ax.text(6.9, 0.35, "Agent G — IETF NMOP (RFC 9940 term ladder)", ha="center", fontsize=10,
            weight="bold", color="#14314f")
    ax.set_title("The observability scenario — a legacy alarm decomposes into the NMOP ladder; "
                 "the ontological trap is that an alarm (a State) is not an anomaly (a deviation)",
                 fontsize=11, y=1.0)
    save(fig, "fig_obs_scene.png")


# --- Figure 2: Act 1 — the ontological cognate, three rungs (reference rescues the middle) -----
def fig_ontology():
    if not loadm("gpt-5.6-sol", 1):
        print("skip fig_obs_ontology: no phase-1 data"); return
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    x = range(len(MODELS)); w = 0.35
    for k, (ref, alpha, hatch) in enumerate([("no-ref", 0.55, ""), ("ref", 1.0, "//")]):
        ys = []
        for M in MODELS:
            inert = [r for r in loadm(M, 1) if r["placement"] in ("one_inert", "both_inert")
                     and r["reference"] == ref]
            ys.append(m(inert, "surviving_false_cognates"))
        xs = [i + (k - 0.5) * w for i in x]
        ax.bar(xs, ys, width=w, color=[COL[M] for M in MODELS], alpha=alpha,
               hatch=hatch, edgecolor="white")
        # explicit value labels so zero-height bars (sol; mini with reference) still read
        for xi, yv in zip(xs, ys):
            ax.text(xi, yv + 0.03, f"{yv:.2f}", ha="center", va="bottom", fontsize=7.8, color="#555")
    ax.set_xticks(list(x)); ax.set_xticklabels([LAB[M] for M in MODELS])
    ax.set_ylabel("alarm↔anomaly cognate survival\n(mean, inert placements)")
    ax.set_ylim(0, 1.08)
    ax.set_title("The ontological cognate, three rungs: sol never takes it; the reference rescues\n"
                 "mini (bar → 0); nano takes it with or without the reference (beyond rescue)",
                 fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#bbb", alpha=0.55, label="no reference (pale)"),
                       Patch(facecolor="#bbb", hatch="//", label="with reference (hatched)")],
              frameon=False, fontsize=8.5, loc="upper right")
    _clean(ax)
    save(fig, "fig_obs_ontology.png")


# --- Figure 3: Act 2 — pragmatics ON/OFF: verdict (capability-gated) vs correlation (robust) ---
def fig_pragmatics():
    if not loadm("gpt-5.6-sol", 3):
        print("skip fig_obs_pragmatics: no phase-3/4 data"); return
    fig, (axV, axC) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    x = range(len(MODELS)); w = 0.35
    # verdict accuracy ON vs OFF
    for k, (prag, alpha) in enumerate([("off", 0.5), ("on", 1.0)]):
        ys = [m([r for r in loadm(M, 3) if r["pragmatics"] == prag], "verdict_accuracy") for M in MODELS]
        axV.bar([i + (k - 0.5) * w for i in x], ys, width=w, color=[COL[M] for M in MODELS],
                alpha=alpha, edgecolor="white")
    axV.set_xticks(list(x)); axV.set_xticklabels([LAB[M] for M in MODELS])
    axV.set_ylabel("verdict accuracy"); axV.set_ylim(0, 1.05)
    axV.set_title("Verdict (act/watch/suppress):\npragmatics pay off — but only if the agent can\n"
                  "carry them (nano: ON ≈ OFF)", fontsize=10)
    # correlation partition-exact rate ON vs OFF
    for k, (prag, alpha) in enumerate([("off", 0.5), ("on", 1.0)]):
        ys = []
        for M in MODELS:
            f = RES / f"obs_config_observability_phase4_{M}.csv"
            rows = [r for r in (list(csv.DictReader(open(f))) if f.exists() else []) if r["pragmatics"] == prag]
            ys.append(sum(1 for r in rows if r["partition_exact"] == "True") / len(rows) if rows else 0.0)
        axC.bar([i + (k - 0.5) * w for i in x], ys, width=w, color=[COL[M] for M in MODELS],
                alpha=alpha, edgecolor="white")
    axC.set_xticks(list(x)); axC.set_xticklabels([LAB[M] for M in MODELS])
    axC.set_ylabel("incidents correlated exactly"); axC.set_ylim(0, 1.05)
    axC.set_title("Correlation (into incidents):\nthe dependency map does the work — every model\n"
                  "correlates ON, all fail OFF", fontsize=10)
    for ax in (axV, axC):
        _clean(ax)
    from matplotlib.patches import Patch
    axV.legend(handles=[Patch(facecolor="#999", alpha=0.5, label="pragmatics OFF (legacy)"),
                        Patch(facecolor="#999", label="pragmatics ON")],
               frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig_obs_pragmatics.png")


if __name__ == "__main__":
    fig_scene()
    fig_ontology()
    fig_pragmatics()
