#!/usr/bin/env python3
"""Memorisation control (Brad's point 7): run the SAME agent on the flagship TAPI/TEAS case
and on its relabelled twin (config_relabel_control, structurally identical, standard identity
stripped) and compare. Similar scores => performance is reasoning from the prompt, not recall
of a TAPI/TEAS mapping seen in training; a drop on the relabelled twin => recall was helping.

First build the twin:  python benchmark/build_relabel_control.py
Then:                  python pipeline/memorisation_control.py            # single agent, sol
                       python pipeline/memorisation_control.py --stack two-agent

Runs both cases at both_cognitive, no-reference and with-reference. Needs OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                        # noqa: E402
from reconcile.metrics import score                                    # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack             # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402

PAIR = [("config_tapi_teas", "recognisable (TAPI/TEAS)"),
        ("config_relabel_control", "relabelled (identity stripped)")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--stack", choices=["single", "two-agent"], default="single")
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    twin = ROOT / "benchmark" / "cases" / "config_relabel_control"
    if not twin.exists():
        print("config_relabel_control not found; run: python benchmark/build_relabel_control.py",
              file=sys.stderr)
        return 2

    def make(use_ref):
        if args.stack == "two-agent":
            return TwoAgentStack(use_reference=use_ref, model=args.model)
        return OpenAIAgentStack(use_reference=use_ref, model=args.model)

    print(f"\nMemorisation control — {args.stack} agent, {args.model}, {args.trials} trial(s)\n")
    print(f"  {'case':26} {'ref':4} {'precision':10} {'resolved':9} {'surv_fc':7}")
    print("  " + "-" * 60)
    rows = []
    for case_name, label in PAIR:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        for use_ref in (False, True):
            ps, rs, fs = [], [], []
            for _ in range(args.trials):
                stack = make(use_ref)
                reference = case.reference if use_ref else None
                try:
                    rec = stack.reconcile(case.model_a, case.model_b,
                                          reference=reference, placement="both_cognitive")
                    s = score(rec, case.gold)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! skip {case_name}/ref={use_ref}: {e}", file=sys.stderr); continue
                ps.append(s.get("precision")); rs.append(s.get("recall"))
                fs.append(s.get("surviving_false_cognates"))
            if ps:
                mp = sum(ps) / len(ps); mr = sum(rs) / len(rs); mf = sum(fs) / len(fs)
                rows.append((label, use_ref, mp, mr, mf))
                print(f"  {label:26} {int(use_ref):<4} {mp:<10.3f} {mr:<9.3f} {mf:<7.2f}")
    print("\nCompare the two cases row for row: if 'relabelled' tracks 'recognisable', the result "
          "is reasoning, not memorised recall of the TAPI/TEAS mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
