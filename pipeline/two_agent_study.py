#!/usr/bin/env python3
"""Two-agent negotiation vs the single-agent shortcut, at both_cognitive.

This answers Brad's point 5 head on: does reconciling with two agents that each see only
their own lifted model and negotiate under real information asymmetry (with bilateral
ratification) reach the same close as one model shown both models at once, and at what cost?

For each schema setting it runs, at both_cognitive, with and without the reference:
  * stack = single   -> OpenAIAgentStack  (one model, both lifts in one call)
  * stack = two-agent -> TwoAgentStack    (two agents, asymmetric, negotiated + ratified)
and scores both against the same validated gold. Quality: precision, resolved fraction,
surviving false cognates, residual. Cost: reasoning tokens, total tokens, and (two-agent
only) the number of negotiation turns.

  # cheap first pass (strong model, one trial):
  python pipeline/two_agent_study.py

  # fuller:
  python pipeline/two_agent_study.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3

Resumable: existing rows in results/two_agent_study.csv are skipped. Needs OPENAI_API_KEY.
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
from reconcile.stacks.agent_openai import OpenAIAgentStack             # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402

# the schema settings the two-agent negotiation applies to (intent uses a separate stack)
CASES = [
    ("config_tapi_teas", "1 · configuration (flagship)"),
    ("config_big_hard", "1 · configuration (hard)"),
    ("config_cross_domain", "3 · cross-domain"),
    ("config_observability", "4 · observability"),
]
# setting labels for the breadth cases, so --cases can address them too
SETTING_OF = {
    "config_tapi_teas": "1 · configuration (flagship)", "config_big_hard": "1 · configuration (hard)",
    "config_cross_domain": "3 · cross-domain", "config_observability": "4 · observability",
    "config_l3vpn": "1 · configuration (breadth)", "config_evpn": "1 · configuration (breadth)",
    "config_xdom_ran": "3 · cross-domain (breadth)", "config_xdom_dc": "3 · cross-domain (breadth)",
    "obs_routing": "4 · observability (breadth)", "obs_compute": "4 · observability (breadth)",
}
COLS = ["case", "setting", "stack", "model", "uses_reference", "trial",
        "precision", "resolved_fraction", "surviving_false_cognates", "residual",
        "turns", "total_tokens", "reasoning_tokens", "latency_s"]


def _row(case_name, setting, stack_label, model, use_ref, trial, rec, scored):
    return {
        "case": case_name, "setting": setting, "stack": stack_label, "model": model,
        "uses_reference": use_ref, "trial": trial,
        "precision": scored.get("precision", ""),
        "resolved_fraction": scored.get("recall", ""),   # internal key is 'recall'
        "surviving_false_cognates": scored.get("surviving_false_cognates", ""),
        "residual": scored.get("residual", ""),
        "turns": rec.work.get("turns", ""),
        "total_tokens": rec.effort.get("total_tokens", ""),
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "latency_s": rec.effort.get("latency_s", ""),
    }


def _done_key(r):
    return (r["case"], r["stack"], r["model"], r["uses_reference"], r["trial"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=6, help="two-agent negotiation round cap")
    ap.add_argument("--ref", choices=["0", "1", "both"], default="both",
                    help="run without the reference (0), with it (1), or both")
    ap.add_argument("--only-stack", choices=["single", "two-agent", "both"], default="both",
                    help="restrict to one stack (e.g. two-agent for a targeted re-run)")
    ap.add_argument("--cases", default="",
                    help="comma-separated case names to run instead of the four core schema cases "
                         "(e.g. the breadth cases config_l3vpn,config_evpn,...)")
    ap.add_argument("--out", default=str(ROOT / "results" / "two_agent_study.csv"))
    args = ap.parse_args()

    ref_values = {"0": (False,), "1": (True,), "both": (False, True)}[args.ref]
    cases = (CASES if not args.cases
             else [(c.strip(), SETTING_OF.get(c.strip(), c.strip()))
                   for c in args.cases.split(",") if c.strip()])

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add(_done_key(r))
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    n = 0
    for case_name, setting in cases:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        for model in models:
            for use_ref in ref_values:
                for t in range(args.trials):
                    stacks = [
                        ("single", OpenAIAgentStack(use_reference=use_ref, model=model)),
                        ("two-agent", TwoAgentStack(use_reference=use_ref, model=model,
                                                    max_rounds=args.max_rounds)),
                    ]
                    if args.only_stack != "both":
                        stacks = [s for s in stacks if s[0] == args.only_stack]
                    for label, stack in stacks:
                        key = (case_name, label, model, str(use_ref), str(t))
                        if key in seen:
                            continue
                        try:
                            reference = case.reference if stack.uses_reference else None
                            rec = stack.reconcile(case.model_a, case.model_b,
                                                  reference=reference, placement="both_cognitive")
                            scored = score(rec, case.gold)
                        except Exception as e:  # noqa: BLE001 - one bad run must not sink the job
                            print(f"  ! skip {case_name}/{label}/{model}/ref={use_ref}/t{t}: {e}",
                                  file=sys.stderr)
                            continue
                        row = _row(case_name, setting, label, model, use_ref, t, rec, scored)
                        w.writerow(row); fh.flush(); n += 1
                        print(f"  {case_name:22} {label:9} {model:12} ref={int(use_ref)} t{t}  "
                              f"P={row['precision']} RF={row['resolved_fraction']} "
                              f"sfc={row['surviving_false_cognates']} "
                              f"tok={row['reasoning_tokens']} turns={row['turns']}",
                              file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Compare, per case/model/ref: does two-agent reach the same precision and resolved "
          "fraction as single, and what does the negotiation cost in reasoning tokens and turns?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
