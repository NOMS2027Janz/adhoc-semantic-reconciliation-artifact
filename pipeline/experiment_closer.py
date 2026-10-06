#!/usr/bin/env python3
"""The capstone's closer: do the TWO ACTS compose? Construct the shared frame, then let the
decisive virtual experiment settle -- reaching the reference's close WITHOUT an authored reference.

The capstone showed the decisive experiment carries two live agents to full commitment in the
settings where a candidate correspondence is already in view (configuration, observability, and
refuting the false cognate). It does NOT lift cross-domain, because there the correspondences
exist only in a CONSTRUCTED shared frame ('treat the transport rate and the service rate as
corresponding'); neither agent's private model holds that frame, so neither surfaces the pair, so
the experiment is never asked. The missing act is CONSTRUCTION -- surfacing the candidate -- which
construct-then-bind already showed closes cross-domain (0.40 -> 0.93).

This run composes the two acts. For each setting, at both_cognitive, it runs four conditions:
  * experiment            -- no frame, WITH experiment (the capstone cell; cross-domain defers)
  * construct+experiment  -- CONSTRUCT the shared frame, inject it into both agents, WITH experiment
  * construct-only        -- the constructed frame, no experiment (isolates construction's own lift)
  * reference             -- the authored reference (the amortised upper bound)

Hypothesis: construct+experiment reaches the reference (~1.00 resolved, 0 surviving false cognates)
with NO authored reference -- the two acts composing, completion demonstrated end to end.

Honest caveat (same as construct-then-bind): construct_reference() reads both models in one pass, a
stand-in for the shared frame two cooperating agents would build together; the per-side binding to
that frame is then done by each agent in-loop, under information asymmetry.

  python pipeline/experiment_closer.py                        # config_cross_domain (the decisive one)
  python pipeline/experiment_closer.py --case config_tapi_teas --trials 2
  python pipeline/experiment_closer.py --all                  # all four settings

Reads resolved fraction, surviving false cognates, experiments spent, constructed entries, tokens.
Resumable. Needs OPENAI_API_KEY.
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
from reconcile.construct import construct_reference                    # noqa: E402
from reconcile.schema_oracle import SchemaOracle                       # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402

ALL_CASES = [
    ("config_cross_domain", "3 · cross-domain"),
    ("config_tapi_teas", "1 · configuration (flagship)"),
    ("config_big_hard", "1 · configuration (hard)"),
    ("config_observability", "4 · observability"),
]
CONDITIONS = ("experiment", "construct+experiment", "construct-only", "reference")
COLS = ["case", "setting", "condition", "model", "trial", "precision", "resolved_fraction",
        "surviving_false_cognates", "residual", "frame_entries", "experiments", "turns",
        "construct_tokens", "total_tokens", "reasoning_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_cross_domain")
    ap.add_argument("--all", action="store_true", help="run all four settings")
    ap.add_argument("--cases", default="",
                    help="comma-separated case names to run (e.g. the breadth cases); overrides --case/--all")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--exp-budget", type=int, default=24)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results" / "experiment_closer.csv"))
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    _label = dict(ALL_CASES)
    if args.cases:
        cases = [(c.strip(), _label.get(c.strip(), c.strip())) for c in args.cases.split(",") if c.strip()]
    elif args.all:
        cases = ALL_CASES
    else:
        cases = [(args.case, _label.get(args.case, args.case))]

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
    for case_name, setting in cases:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        for t in range(args.trials):
            # construct the shared frame ONCE per case/trial, reused by the frame conditions.
            frame = None
            frame_effort = {}
            need_frame = any((case_name, c, args.model, str(t)) not in seen
                             for c in ("construct+experiment", "construct-only"))
            if need_frame:
                frame, frame_effort = construct_reference(case.model_a, case.model_b, args.model)
                print(f"  {case_name}: constructed frame of {frame_effort.get('entries')} entries "
                      f"({frame_effort.get('construct_total_tokens')} tok)", file=sys.stderr)

            def build(condition):
                if condition == "experiment":
                    return TwoAgentStack(use_reference=False, model=args.model,
                                         max_rounds=args.max_rounds,
                                         oracle=SchemaOracle(case, budget=args.exp_budget))
                if condition == "construct+experiment":
                    return TwoAgentStack(use_reference=False, model=args.model,
                                         max_rounds=args.max_rounds, shared_frame=frame,
                                         oracle=SchemaOracle(case, budget=args.exp_budget))
                if condition == "construct-only":
                    return TwoAgentStack(use_reference=False, model=args.model,
                                         max_rounds=args.max_rounds, shared_frame=frame)
                return TwoAgentStack(use_reference=True, model=args.model,
                                     max_rounds=args.max_rounds)  # reference

            for condition in CONDITIONS:
                if (case_name, condition, args.model, str(t)) in seen:
                    continue
                stack = build(condition)
                reference = case.reference if stack.uses_reference else None
                try:
                    rec = stack.reconcile(case.model_a, case.model_b,
                                          reference=reference, placement="both_cognitive")
                    s = score(rec, case.gold)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! skip {case_name}/{condition}/t{t}: {e}", file=sys.stderr)
                    continue
                uses_frame = condition in ("construct+experiment", "construct-only")
                row = {
                    "case": case_name, "setting": setting, "condition": condition,
                    "model": args.model, "trial": t,
                    "precision": s.get("precision", ""), "resolved_fraction": s.get("recall", ""),
                    "surviving_false_cognates": s.get("surviving_false_cognates", ""),
                    "residual": s.get("residual", ""),
                    "frame_entries": (frame_effort.get("entries", "") if uses_frame else ""),
                    "experiments": rec.work.get("experiments", 0),
                    "turns": rec.work.get("turns", ""),
                    "construct_tokens": (frame_effort.get("construct_total_tokens", "")
                                         if uses_frame else ""),
                    "total_tokens": rec.effort.get("total_tokens", ""),
                    "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                }
                w.writerow(row); fh.flush(); n += 1
                print(f"  {case_name:20} {condition:20} P={row['precision']} "
                      f"RF={row['resolved_fraction']} sfc={row['surviving_false_cognates']} "
                      f"frame={row['frame_entries']} exp={row['experiments']}", file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Read cross_domain: does 'construct+experiment' reach 'reference' (~1.00, 0 surviving false "
          "cognates) while 'experiment' alone stays ~0.20? If so, the two acts compose: construction "
          "surfaces the candidates, the experiment settles them, and completion is shown end to end "
          "with no authored reference. Compare 'construct-only' to see how much the experiment adds "
          "on top of the frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
