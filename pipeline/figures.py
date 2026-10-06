#!/usr/bin/env python3
"""Generate the report figures from a results CSV (and the scaling CSV).

    python figures.py --case config_big_hard        # reads results/config_big_hard.csv
    python figures.py --csv results/archive/xyz.csv
    python figures.py --scaling                      # scaling figure only

Writes PNGs to figures/. Light, committed look; colourblind-safe categorical
palette (dataviz reference palette, slots 1-3); every bar carries a direct value
label so the aqua-contrast relief rule is satisfied.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"

# dataviz reference palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
PLACE_LABEL = {"both_cognitive": "both\ncognitive", "one_inert": "one\ninert", "both_inert": "both\ninert"}
MODEL_ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
MODEL_LABEL = {"gpt-5.6-sol": "sol (strong)", "gpt-5-mini": "mini", "gpt-5-nano": "nano (weak)"}
MODEL_COLOR = {"gpt-5.6-sol": BLUE, "gpt-5-mini": ORANGE, "gpt-5-nano": AQUA}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def _load(csv_path):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["stack"].startswith("openai")]
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    def mean(rs, k):
        xs = [num(r[k]) for r in rs if num(r[k]) is not None]
        return sum(xs) / len(xs) if xs else None
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    models += sorted({r["model"] for r in rows} - set(models))
    placements = [p for p in PLACEMENTS if any(r["placement"] == p for r in rows)]
    agg = {}
    for m in models:
        for p in placements:
            for ref in ("False", "True"):
                rs = [r for r in rows if r["model"] == m and r["placement"] == p and r["uses_reference"] == ref]
                if rs:
                    agg[(m, p, ref)] = {k: mean(rs, k) for k in
                                        ("recall", "precision", "reasoning_tokens", "total_tokens", "latency_s",
                                         "surviving_false_cognates", "residual")}
    return models, placements, agg


def _save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


def fig_effort(models, placements, agg, model):
    """Grouped bars: reasoning tokens with vs without reference, by placement, one model."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    x = range(len(placements))
    w = 0.38
    no = [agg.get((model, p, "False"), {}).get("reasoning_tokens") or 0 for p in placements]
    ye = [agg.get((model, p, "True"), {}).get("reasoning_tokens") or 0 for p in placements]
    b1 = ax.bar([i - w / 2 for i in x], no, w, label="without reference", color=ORANGE)
    b2 = ax.bar([i + w / 2 for i in x], ye, w, label="with reference", color=BLUE)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([PLACE_LABEL[p] for p in placements])
    ax.set_ylabel("reasoning tokens (mean)", color=INK2, fontsize=9)
    ax.set_title(f"The reference cuts deliberation, most steeply where a single agent reconstructs — {MODEL_LABEL.get(model, model)}",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, "fig_effort_substitution.png")


def fig_gradient(models, placements, agg):
    """Line: reference benefit (reasoning tokens saved) across the cognition spectrum, per model."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    _style(ax)
    ax.axhline(0, color=BASE, linewidth=1)
    x = range(len(placements))
    # sol and mini converge, so their reference benefit is a meaningful effort measurement; nano
    # does not converge at both-cognitive (it hits the round cap either way, ~85-110k tokens), so its
    # "benefit" there is a difference of two non-converged runs, not a clean value — omitted.
    shown = [m for m in models if m in ("gpt-5.6-sol", "gpt-5-mini")]
    for si, m in enumerate(shown):
        y = []
        for p in placements:
            no = agg.get((m, p, "False"), {}).get("reasoning_tokens")
            ye = agg.get((m, p, "True"), {}).get("reasoning_tokens")
            y.append(None if no is None or ye is None else no - ye)
        ax.plot(list(x), y, marker="o", markersize=6, linewidth=2, color=MODEL_COLOR.get(m, MUTED),
                label=MODEL_LABEL.get(m, m))
        for xi, yi in zip(x, y):
            if yi is None:
                continue
            off = 11 if m == "gpt-5.6-sol" else -15
            ax.annotate(f"{yi:+.0f}", (xi, yi), textcoords="offset points", xytext=(0, off),
                        ha="center", va="bottom", fontsize=8, color=MODEL_COLOR.get(m, INK2),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(list(x))
    ax.set_xticklabels([PLACE_LABEL[p] for p in placements])
    ax.set_ylabel("reasoning tokens saved by the reference", color=INK2, fontsize=9)
    ax.set_title("Reference benefit across the cognition spectrum (sol, mini)", color=INK, fontsize=11, loc="left")
    ax.annotate("nano does not converge either way at both-cognitive\n"
                "(round cap, ~85-110k tokens); its benefit is not a clean value",
                xy=(0.71, 0.56), xycoords="axes fraction", ha="center", va="center",
                fontsize=7.6, color=MUTED, style="italic")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, "fig_cognition_gradient.png")


def fig_quality(models, placements, agg):
    """Grouped bars: precision with vs without reference, by model, averaged over placements.

    On the hard case the traps show up as false positives, so precision is where the
    reference separates quality.
    """
    # the reference's error-prevention shows where a single agent reconstructs an inert side;
    # at both-cognitive bilateral ratification already holds precision, so average over inert only.
    inert = [p for p in placements if p != "both_cognitive"]

    def avg_prec(model, ref):
        vals = [agg[(model, p, ref)]["precision"] for p in inert
                if (model, p, ref) in agg and agg[(model, p, ref)]["precision"] is not None]
        return sum(vals) / len(vals) if vals else 0
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _style(ax)
    x = range(len(models))
    w = 0.38
    no = [avg_prec(m, "False") for m in models]
    ye = [avg_prec(m, "True") for m in models]
    b1 = ax.bar([i - w / 2 for i in x], no, w, label="without reference", color=ORANGE)
    b2 = ax.bar([i + w / 2 for i in x], ye, w, label="with reference", color=BLUE)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.28)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models])
    ax.set_ylabel("precision (mean over inert placements)", color=INK2, fontsize=9)
    ax.set_title("Precision at the inert placements, with vs without the reference — the harder case",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    _save(fig, "fig_quality.png")


def fig_verify(case, model="gpt-5.6-sol"):
    """Post-verification recall across placements, without vs with reference.

    Verification clears the false positives everywhere (precision -> ~1); the story
    is recall: without the reference, a verify-and-repair that cannot confirm a
    correspondence discards it, and the loss grows as cognition recedes. The
    reference supplies the confirmation, so recall holds.
    """
    path = ROOT / "results" / f"verify_{case}.csv"
    if not path.exists():
        print(f"no verify CSV at {path}")
        return
    rows = [r for r in csv.DictReader(open(path)) if r["model"] == model and r["stage"] == "post"]
    if not rows:
        print("no post rows for", model)
        return
    # the verify-and-repair pass is the inert-placement check (at both-cognitive the two agents
    # confirm through ratification and the decisive experiment, not this single-pass repair).
    placements = [p for p in PLACEMENTS if p != "both_cognitive" and any(r["placement"] == p for r in rows)]
    def mean(pl, ref):
        xs = [float(r["recall"]) for r in rows if r["placement"] == pl and r["uses_reference"] == ref]
        return sum(xs) / len(xs) if xs else None
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _style(ax)
    x = range(len(placements))
    no = [mean(p, "False") for p in placements]
    ye = [mean(p, "True") for p in placements]
    ax.plot(list(x), no, marker="o", markersize=6, linewidth=2, color=ORANGE, label="without reference")
    ax.plot(list(x), ye, marker="o", markersize=6, linewidth=2, color=BLUE, label="with reference")
    for xi, v in zip(x, no):
        if v is not None:
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK2)
    for xi, v in zip(x, ye):
        if v is not None:
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0.5, 1.06)
    ax.set_xticks(list(x))
    ax.set_xticklabels([PLACE_LABEL[p] for p in placements])
    ax.set_ylabel("resolved fraction after verify-and-repair", color=INK2, fontsize=9)
    ax.set_title("At the inert placements, the reference preserves the resolved fraction as cognition recedes",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "fig_verify.png")


def fig_scaling():
    path = ROOT / "results" / "scaling_scaling_otn.csv"
    if not path.exists():
        print("no scaling CSV; run: python scaling.py --write")
        return
    rows = list(csv.DictReader(open(path)))
    N = [int(r["N"]) for r in rows]
    with_ref = [int(r["with_ref_ops"]) for r in rows]
    without = [int(r["without_ref_ops"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    _style(ax)
    ax.plot(N, without, marker="o", markersize=5, linewidth=2, color=ORANGE,
            label="align every pair (no reference) — ~N(N−1)/2")
    ax.plot(N, with_ref, marker="o", markersize=5, linewidth=2, color=BLUE,
            label="bind once (with reference) — ~N")
    ax.annotate(f"{without[-1]}", (N[-1], without[-1]), textcoords="offset points", xytext=(-4, 6),
                ha="right", fontsize=9, color=INK2)
    ax.annotate(f"{with_ref[-1]}", (N[-1], with_ref[-1]), textcoords="offset points", xytext=(-4, 6),
                ha="right", fontsize=9, color=INK2)
    ax.set_xlabel("number of systems N", color=INK2, fontsize=9)
    ax.set_ylabel("reconciliation operations", color=INK2, fontsize=9)
    ax.set_title("Work grows with N given a reference, with N² without", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, "fig_scaling.png")


# The both-cognitive cells are REPLACED by the real two-agent negotiation, so that the
# both-cognitive point in every figure is the two-agent result; one-inert / both-inert stay
# single-agent (a lone reconstructing agent). Read from results/two_agent_full.csv at figure
# time rather than hardcoded, so the figures cannot drift from the data after a re-run.
def _two_agent_bothcog(case):
    path = ROOT / "results" / "two_agent_full.csv"
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(open(path)) if r.get("case") == case]

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, *keys):
        for k in keys:                      # published column is resolved_fraction;
            xs = [num(r[k]) for r in rs     # older files emit it as recall
                  if k in r and num(r[k]) is not None]
            if xs:
                return sum(xs) / len(xs)
        return None

    out = {}
    for m in {r["model"] for r in rows}:
        for ref in ("False", "True"):
            rs = [r for r in rows if r["model"] == m and r["uses_reference"] == ref]
            if rs:
                out[(m, ref)] = dict(
                    recall=mean(rs, "resolved_fraction", "recall"),
                    precision=mean(rs, "precision"),
                    reasoning_tokens=mean(rs, "reasoning_tokens"),
                )
    return out


def _override_both_cognitive(agg, case):
    for (m, ref), vals in _two_agent_bothcog(case).items():
        cell = agg.get((m, "both_cognitive", ref))
        if cell is not None:
            cell.update({k: v for k, v in vals.items() if v is not None})
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="config_big_hard")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--scaling", action="store_true", help="scaling figure only")
    args = ap.parse_args()
    if args.scaling:
        fig_scaling()
        return 0
    csv_path = Path(args.csv) if args.csv else ROOT / "results" / f"{args.case}.csv"
    if not Path(csv_path).exists():
        print(f"no results at {csv_path}")
        return 1
    models, placements, agg = _load(csv_path)
    agg = _override_both_cognitive(agg, args.case)
    strong = models[0]
    fig_effort(models, placements, agg, strong)
    fig_gradient(models, placements, agg)
    fig_quality(models, placements, agg)
    fig_verify(args.case)
    fig_scaling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
