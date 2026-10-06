#!/usr/bin/env python3
"""Dump the full two-agent transcript for ONE case WITH the decisive virtual experiment in play,
to see the completion happen: which candidate correspondences the agents choose to PROVISION,
what the virtual network reads back (confirmed / false-cognate / no-counterpart), and how those
verdicts settle what dialogue alone would have deferred.

This is the capstone's companion to `two_agent_transcript.py`. That one shows two agents talking
and honestly deferring the granularity-divergent pairs without a reference. This one shows the
same two agents, no reference, but now able to run the experiment -- and it captures the moment a
deferral becomes a decision: the agents operate a candidate in virtual space, read the invariants
back, and bind or refute it. The transcript itself is the result: machine cognition closing a
correspondence by operation, not by assertion.

  python pipeline/experiment_transcript.py                        # config_tapi_teas, no reference
  python pipeline/experiment_transcript.py --case config_cross_domain   # the 'grade' false-cognate case
  python pipeline/experiment_transcript.py --exp-budget 24 --max-rounds 6

Writes results/experiment_transcript_<case>.json and prints a per-turn summary with the
experiment provisions and verdicts inline. No reference (the whole point). Needs OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_tapi_teas")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--exp-budget", type=int, default=24,
                    help="max decisive experiments (keeps agents from brute-forcing every pair)")
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    oracle = SchemaOracle(case, budget=args.exp_budget)
    stack = TwoAgentStack(use_reference=False, model=args.model,
                          max_rounds=args.max_rounds, oracle=oracle)
    rec = stack.reconcile(case.model_a, case.model_b, reference=None, placement="both_cognitive")
    transcript = getattr(stack, "last_transcript", [])
    s = score(rec, case.gold)

    out = ROOT / "results" / f"experiment_transcript_{args.case}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "case": args.case, "model": args.model, "reference": False, "with_experiment": True,
        "precision": s.get("precision", ""), "resolved_fraction": s.get("recall", ""),
        "surviving_false_cognates": s.get("surviving_false_cognates", ""),
        "confirmed": [sorted(p) for p in rec.proposed],
        "residual_a": rec.residual_a, "residual_b": rec.residual_b,
        "experiments_spent": rec.work.get("experiments", 0),
        "experiment_log": [{"a": r.a_id, "b": r.b_id, "ok": r.ok,
                            "confirmed": r.confirmed, "relation": r.relation} for r in oracle.log],
        "work": rec.work, "effort": rec.effort, "transcript": transcript,
    }, indent=2))

    print(f"\n=== {args.case} (no-ref, WITH experiment) — {len(transcript)} turns, "
          f"{len(rec.proposed)} confirmed, {rec.work.get('experiments', 0)} experiments spent, "
          f"P={s.get('precision')} RF={s.get('recall')} sFC={s.get('surviving_false_cognates')} ===\n")
    for i, turn in enumerate(transcript):
        side = turn["from"]
        q = turn.get("questions", []); a = turn.get("answers", [])
        p = turn.get("proposals", []); r = turn.get("ratifications", [])
        ex = turn.get("experiments", []); xr = turn.get("experiment_results", [])
        nc = turn.get("no_counterpart", [])
        print(f"[{i+1}] Agent {side}  done={turn.get('done')}")
        for x in q:
            print(f"      ASK   {x['their_id']}: {x['ask'][:80]}")
        for x in a:
            print(f"      ANSW  {x['my_id']}: {x.get('gloss','')[:78]}")
        for x in p:
            print(f"      PROP  {x['my_id']} <-> {x['their_id']}  conf={x['confidence']}  {x['rationale'][:66]}")
        for x in ex:
            print(f"      EXPT  provision {x['my_id']} <-> {x['their_id']}?  {x.get('question','')[:56]}")
        for x in xr:
            mark = "CONFIRM" if x['confirmed'] else ("OVER-BUDGET" if not x['ok'] else "REFUTE ")
            print(f"      >>>   {mark} {x['a']} <-> {x['b']}   [{x['relation']}]")
        for x in r:
            verdict = "ACCEPT" if x['accept'] else "REJECT"
            print(f"      RTFY  {verdict} {x['their_id']}<->{x['my_id']}  {x['reason'][:56]}")
        if nc:
            print(f"      NO-COUNTERPART {nc}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("Read: look for EXPT (an agent chooses to provision a candidate) followed by >>> "
          "(the virtual network's decisive verdict). A CONFIRM binds a pair dialogue would have "
          "deferred; a REFUTE kills a false cognate by operation. That sequence IS the completion "
          "the both-cognitive claim promises -- shown, not argued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
