#!/usr/bin/env python3
"""Summarise benchmark/leaderboard.csv into an agent-versus-controls comparison.

Averages the agent's trials and lays every stack side by side per case, so the
value the lift and the reference add over the baseline (a prior-art-style lexical
matcher) is explicit. Offline; reads only the CSV.

  python benchmark/leaderboard_summary.py                 # print the table
  python benchmark/leaderboard_summary.py --placement both_cognitive
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = ("precision", "resolved_fraction", "f1", "surviving_false_cognates", "residual")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def summarise(rows, placement_filter=None):
    # key: (case, stack, model, placement) -> {metric: [values]}
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pl = r.get("placement", "")
        is_control = pl in ("", "n/a")
        if placement_filter and not is_control and pl != placement_filter:
            continue
        key = (r.get("case", ""), r.get("stack", ""), r.get("model", "-"), pl)
        for m in NUM:
            v = _f(r.get(m))
            if v is not None:
                agg[key][m].append(v)
    out = {}
    for key, mv in agg.items():
        out[key] = {m: (round(st.mean(vs), 3) if vs else None) for m, vs in mv.items()}
        out[key]["n"] = max((len(vs) for vs in mv.values()), default=0)
    return out


STACK_ORDER = ["baseline (no reference)", "reference control",
               "agent (no-ref)", "agent (ref)"]


def stack_rank(stack):
    for i, s in enumerate(STACK_ORDER):
        if stack.startswith(s.split(" (")[0]) and stack == s:
            return i
    return 9


def render(summary):
    by_case = defaultdict(list)
    for (case, stack, model, pl), mv in summary.items():
        by_case[case].append((stack, model, pl, mv))
    cols = ["stack", "model", "placement", "precision", "resolved", "f1",
            "surv_fc", "residual", "n"]
    keymap = {"surv_fc": "surviving_false_cognates", "resolved": "resolved_fraction"}
    lines = []
    for case in sorted(by_case):
        lines.append(f"\n=== {case} ===")
        lines.append("  ".join(c.ljust(11 if c in ("stack",) else 8) for c in cols))
        entries = sorted(by_case[case], key=lambda e: (stack_rank(e[0]), e[1], e[2]))
        for stack, model, pl, mv in entries:
            cells = []
            for c in cols:
                if c == "stack":
                    cells.append(stack.ljust(11))
                elif c == "model":
                    cells.append(str(model).ljust(8))
                elif c == "placement":
                    cells.append((pl or "n/a").ljust(8))
                elif c == "n":
                    cells.append(str(mv.get("n", "")).ljust(8))
                else:
                    v = mv.get(keymap.get(c, c))
                    cells.append(("" if v is None else f"{v:g}").ljust(8))
            lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "leaderboard.csv"))
    ap.add_argument("--placement", default=None,
                    help="restrict agent rows to one placement (controls always shown)")
    args = ap.parse_args()
    path = Path(args.csv)
    if not path.exists():
        print(f"no {path.name}; run benchmark/leaderboard.py first")
        return 2
    rows = load(path)
    print(render(summarise(rows, args.placement)))
    n_agent = sum(1 for r in rows if r.get("stack", "").startswith("agent"))
    print(f"\n{len(rows)} rows ({n_agent} agent). Controls average over 1 run; agent over trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
