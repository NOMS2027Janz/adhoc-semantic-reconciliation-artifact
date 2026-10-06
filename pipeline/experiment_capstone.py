#!/usr/bin/env python3
"""The completion capstone: does the DECISIVE VIRTUAL EXPERIMENT carry two live agents to full
commitment WITHOUT a shared reference?

The theory says the fully-cognitive case is complete in principle because two live agents can
provision a candidate correspondence in virtual space, operate it, read it back, and check the
invariants -- so any correspondence question is settled, not deferred. Dialogue alone does not do
this (it defers), and a shared reference amortises it. This run adds the missing corner: two agents
that may run the experiment (via the SchemaOracle -- the virtual network's ground truth) and shows
they reach full commitment and refute the false cognates by OPERATION, with no reference.

For each schema setting, at both_cognitive, it runs three conditions:
  * dialogue        -- two agents, no reference, no experiment (defers)
  * experiment      -- two agents, no reference, WITH the decisive virtual experiment
  * reference       -- two agents, WITH the shared reference, no experiment (the amortised version)

  python pipeline/experiment_capstone.py                 # sol
  python pipeline/experiment_capstone.py --model gpt-5.6-sol --exp-budget 24 --trials 2

Reads resolved fraction, surviving false cognates, experiments spent, turns, tokens. Resumable.
Needs OPENAI_API_KEY. Expected: experiment matches reference at ~1.00 resolved and 0 surviving
false cognates, WITHOUT the reference -- completion demonstrated, not just argued.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                        # noqa: E402
from reconcile.metrics import score                                    # noqa: E402
from reconcile.schema_oracle import SchemaOracle                       # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402

CASES = [
    ("config_tapi_teas", "1 · configuration (flagship)"),
    ("config_big_hard", "1 · configuration (hard)"),
    ("config_cross_domain", "3 · cross-domain"),
    ("config_observability", "4 · observability"),
]
COLS = ["case", "setting", "condition", "model", "trial", "precision", "resolved_fraction",
        "surviving_false_cognates", "residual", "experiments", "turns",
        "total_tokens", "reasoning_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--exp-budget", type=int, default=24,
                    help="max decisive experiments per reconciliation (keeps it from brute-forcing)")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "experiment_capstone.csv"))
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add((r["case"], r["condition"], r["model"], r["trial"]))
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    n = 0
    for case_name, setting in CASES:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        for t in range(args.trials):
            def build(condition):
                if condition == "dialogue":
                    return TwoAgentStack(use_reference=False, model=args.model,
                                         max_rounds=args.max_rounds), None
                if condition == "experiment":
                    oracle = SchemaOracle(case, budget=args.exp_budget)
                    return TwoAgentStack(use_reference=False, model=args.model,
                                         max_rounds=args.max_rounds, oracle=oracle), oracle
                return TwoAgentStack(use_reference=True, model=args.model,
                                     max_rounds=args.max_rounds), None  # reference

            for condition in ("dialogue", "experiment", "reference"):
                if (case_name, condition, args.model, str(t)) in seen:
                    continue
                stack, _ = build(condition)
                reference = case.reference if stack.uses_reference else None
                try:
                    rec = stack.reconcile(case.model_a, case.model_b,
                                          reference=reference, placement="both_cognitive")
                    s = score(rec, case.gold)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! skip {case_name}/{condition}/t{t}: {e}", file=sys.stderr)
                    continue
                row = {
                    "case": case_name, "setting": setting, "condition": condition,
                    "model": args.model, "trial": t,
                    "precision": s.get("precision", ""), "resolved_fraction": s.get("recall", ""),
                    "surviving_false_cognates": s.get("surviving_false_cognates", ""),
                    "residual": s.get("residual", ""),
                    "experiments": rec.work.get("experiments", 0),
                    "turns": rec.work.get("turns", ""),
                    "total_tokens": rec.effort.get("total_tokens", ""),
                    "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                }
                w.writerow(row); fh.flush(); n += 1
                print(f"  {case_name:22} {condition:10} P={row['precision']} "
                      f"RF={row['resolved_fraction']} sfc={row['surviving_false_cognates']} "
                      f"exp={row['experiments']} turns={row['turns']}", file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Read: does 'experiment' reach the same resolved fraction as 'reference' (~1.00) with 0 "
          "surviving false cognates, WITHOUT a reference? If so, the both-cognitive completion claim "
          "is demonstrated: the decisive experiment closes what dialogue alone defers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
