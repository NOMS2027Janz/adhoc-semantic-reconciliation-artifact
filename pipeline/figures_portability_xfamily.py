#!/usr/bin/env python3
"""Cross-family portability: is a lifted model comprehensible OUTSIDE the family that lifted it?

The in-family portability run (comprehension_portability.csv) shows a solid lift is comprehensible to
the GPT consumer ladder (sol / mini / nano). The obvious objection is that a GPT consumer merely reads
a GPT lifter's output. This figure answers it: it places foreign-family consumers next to the in-family
ladder on the SAME comprehension instrument, with the JUDGE pinned to the sol gateway throughout so the
only thing that changes is who does the comprehending.

Consumers plotted, strong -> weak, in-family first then cross-family:
    gpt-5.6-sol, gpt-5-mini, gpt-5-nano   |   deepseek-chat (foreign strong), qwen2.5:7b (open weak)

Two panels, each bare vs anchored, averaged over cases and trials:
    left  -- meaning score        (portability: can it say what each concept denotes)
    right -- confabulation rate    (the hazard: asserting a meaning the key does not support)

Reads results/comprehension_portability.csv (+ _xfamily.csv). Either may be absent; only present
consumers are drawn. Writes figures/fig_portability_xfamily.png. Light background, no em-dashes.
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
CSV_IN = ROOT / "results" / "comprehension_portability.csv"
CSV_XF = ROOT / "results" / "comprehension_portability_xfamily.csv"
NAVY, INK, SURFACE, FAINT = "#14314f", "#1a1a1a", "#fbfaf8", "#6a655e"
TEAL, RED, ORANGE = "#1baf7a", "#b23a48", "#eb6834"

# strong -> weak; in-family then cross-family. Display label + which "family band" it sits in.
ORDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano", "deepseek-chat", "qwen2.5:7b"]
LABEL = {"gpt-5.6-sol": "5.6\nSol", "gpt-5-mini": "5\nMini", "gpt-5-nano": "5\nNano",
         "deepseek-chat": "DeepSeek", "qwen2.5:7b": "Qwen\n7B"}
XFAMILY = {"deepseek-chat", "qwen2.5:7b"}


def _load():
    rows = []
    for path in (CSV_IN, CSV_XF):
        if path.exists():
            rows += list(csv.DictReader(path.open()))
    return rows


def _mean(rows, consumer, arm, col):
    v = [float(r[col]) for r in rows
         if r["consumer"] == consumer and r["arm"] == arm
         and r.get(col) not in ("", "None", None)]
    return sum(v) / len(v) if v else float("nan")


def main():
    rows = _load()
    present = [c for c in ORDER if any(r["consumer"] == c for r in rows)]
    if not present:
        print("no portability rows found yet (need comprehension_portability[_xfamily].csv)")
        return

    x = np.arange(len(present))
    w = 0.38
    bare_m = [_mean(rows, c, "bare", "meaning_score") for c in present]
    anch_m = [_mean(rows, c, "anchored", "meaning_score") for c in present]
    bare_c = [_mean(rows, c, "bare", "confabulation_rate") for c in present]
    anch_c = [_mean(rows, c, "anchored", "confabulation_rate") for c in present]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    def shade_xfamily(ax):
        idx = [i for i, c in enumerate(present) if c in XFAMILY]
        if idx:
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color=ORANGE, alpha=0.06, zorder=0)

    # left: meaning score, bare vs anchored
    shade_xfamily(axL)
    axL.bar(x - w / 2, bare_m, w, color=FAINT, label="bare", zorder=3)
    axL.bar(x + w / 2, anch_m, w, color=TEAL, label="anchored (+ reference)", zorder=3)
    axL.set_title("Meaning score: the lift is comprehensible across families",
                  fontsize=11, color=NAVY)
    axL.set_ylim(0.0, 1.05)
    axL.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.02, 0.02))

    # right: confabulation rate, bare vs anchored
    shade_xfamily(axR)
    axR.bar(x - w / 2, bare_c, w, color=FAINT, label="bare", zorder=3)
    axR.bar(x + w / 2, anch_c, w, color=RED, label="anchored (+ reference)", zorder=3)
    axR.set_title("Confabulation rate: the hazard, and how anchoring moves it",
                  fontsize=11, color=NAVY)
    axR.set_ylim(0.0, max(0.4, np.nanmax(bare_c + anch_c + [0.0]) * 1.25))
    axR.legend(frameon=False, fontsize=9.5, loc="upper left")

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL.get(c, c) for c in present], fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Portability is not a within-family artefact "
                 "(in-family ladder | shaded: cross-family consumers, sol judge fixed)",
                 fontsize=12.5, color=NAVY, y=1.0)
    fig.text(0.5, -0.02,
             "A foreign STRONG reasoner (DeepSeek) matches the in-family strong tier; the open WEAK model "
             "(Qwen) comprehends far less and confabulates, and the reference only partly rescues it. "
             "Portability tracks reasoning, not family. Judge held fixed throughout.",
             ha="center", fontsize=9.5, color=ORANGE, style="italic")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "fig_portability_xfamily.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")

    # console table
    print("wrote", out.name)
    print(f"  {'consumer':14}{'meaning bare':14}{'meaning anch':14}{'confab bare':13}{'confab anch':12}")
    for c, bm, am, bc, ac in zip(present, bare_m, anch_m, bare_c, anch_c):
        print(f"  {c:14}{bm:<14.2f}{am:<14.2f}{bc:<13.2f}{ac:<12.2f}")


if __name__ == "__main__":
    main()
