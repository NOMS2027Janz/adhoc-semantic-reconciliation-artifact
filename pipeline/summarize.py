#!/usr/bin/env python3
"""Summarise a results CSV: per-model, per-placement, per-condition averages over
trials, the with-vs-without-reference deltas, and a benefit matrix that shows
whether the reference's benefit grows as cognition recedes (across placements) and
as the model weakens (across models).

    python summarize.py                 # summarises results/config_tapi_teas.csv
    python summarize.py --case <name>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS = [
    ("precision", "precision", "{:.2f}"),
    ("recall", "recall", "{:.2f}"),
    ("surviving_false_cognates", "false cognates", "{:.1f}"),
    ("residual", "residual", "{:.1f}"),
    ("total_tokens", "total tokens", "{:.0f}"),
    ("reasoning_tokens", "reasoning tokens", "{:.0f}"),
    ("latency_s", "latency (s)", "{:.1f}"),
]
PLACEMENT_ORDER = ["both_cognitive", "one_inert", "both_inert"]


def _avg(rows, key):
    xs = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return sum(xs) / len(xs) if xs else None


def _means(rows):
    return {k: _avg(rows, k) for k, _, _ in FIELDS}


def _ordered(values, order):
    present = [v for v in order if v in values]
    return present + sorted(values - set(present))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="config_tapi_teas")
    args = ap.parse_args()

    path = ROOT / "results" / f"{args.case}.csv"
    if not path.exists():
        print(f"No results at {path}. Run: python run.py --agent --trials 3")
        return 1
    rows = list(csv.DictReader(path.open()))
    agent = [r for r in rows if r["stack"].startswith("openai")]

    print(f"\nCase: {args.case}\n")
    for r in rows:
        if not r["stack"].startswith("openai"):
            print(f"  {r['stack']:<24} precision {r['precision']}  recall {r['recall']}  "
                  f"false_cog {r['surviving_false_cognates']}")
    if not agent:
        print("\n(no agent runs yet: python run.py --agent --trials 3)")
        return 0

    models = _ordered({r["model"] for r in agent}, [])
    placements = _ordered({r["placement"] for r in agent}, PLACEMENT_ORDER)

    # benefit[(model, placement)] = {field: with - without}
    benefit: dict[tuple, dict] = {}
    for model in models:
        print(f"\n########## model: {model} ##########")
        for pl in placements:
            prows = [r for r in agent if r["model"] == model and r["placement"] == pl]
            if not prows:
                continue
            print(f"\n  === placement: {pl} ===")
            m = {}
            for label, ref in [("WITHOUT reference", "False"), ("WITH reference", "True")]:
                rs = [r for r in prows if r["uses_reference"] == ref]
                m[ref] = _means(rs)
                print(f"    {label}   (n={len(rs)})")
                for key, name, fmt in FIELDS:
                    v = m[ref][key]
                    if v is not None:
                        print(f"       {name:<18}{fmt.format(v)}")
            d = {}
            print("    DELTA (with reference - without):")
            for key, name, _ in FIELDS:
                a, b = m["True"][key], m["False"][key]
                if a is not None and b is not None:
                    d[key] = a - b
                    print(f"       {name:<18}" + (f"{d[key]:+.1f}" if abs(d[key]) >= 1 else f"{d[key]:+.2f}"))
            benefit[(model, pl)] = d

    # benefit matrix: rows = models (weaker downward, if you pass them that way), cols = placements
    def matrix(key, name, fmt, sign=1):
        print(f"\n  {name}   (positive = reference helps more)")
        print("    " + "model \\ placement".ljust(24) + "".join(pl.ljust(16) for pl in placements))
        for model in models:
            cells = []
            for pl in placements:
                d = benefit.get((model, pl), {}).get(key)
                cells.append((fmt.format(sign * d)).ljust(16) if d is not None else "-".ljust(16))
            print("    " + str(model).ljust(24) + "".join(cells))

    print("\n\n=== Reference benefit matrix ===")
    print("Across a row: does the benefit grow as cognition recedes?")
    print("Down a column: does the benefit grow as the model weakens?")
    matrix("reasoning_tokens", "reasoning tokens saved", "{:+.0f}", sign=-1)
    matrix("recall", "recall gain", "{:+.2f}", sign=1)
    matrix("latency_s", "latency saved (s)", "{:+.1f}", sign=-1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
