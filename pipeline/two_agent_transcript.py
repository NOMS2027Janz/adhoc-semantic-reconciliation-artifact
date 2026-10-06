#!/usr/bin/env python3
"""Dump the full two-agent negotiation transcript for ONE case, to see WHAT the agents
actually did -- did they interrogate the other and then defer (genuine caution), or defer
without asking enough (a protocol issue)? Settles whether the low no-reference resolved
fraction is a real finding or an artefact of surface-only advertisement.

  python pipeline/two_agent_transcript.py                       # config_tapi_teas, no reference
  python pipeline/two_agent_transcript.py --case config_cross_domain
  python pipeline/two_agent_transcript.py --reference           # with the shared reference

Writes results/transcript_<case>_<ref|noref>.json and prints a per-turn summary:
questions asked, answers given, proposals made, and ratifications. Needs OPENAI_API_KEY.
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
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_tapi_teas")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--reference", action="store_true", help="run WITH the shared reference")
    ap.add_argument("--max-rounds", type=int, default=8)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    stack = TwoAgentStack(use_reference=args.reference, model=args.model, max_rounds=args.max_rounds)
    reference = case.reference if args.reference else None
    rec = stack.reconcile(case.model_a, case.model_b, reference=reference, placement="both_cognitive")
    transcript = getattr(stack, "last_transcript", [])

    tag = "ref" if args.reference else "noref"
    out = ROOT / "results" / f"transcript_{args.case}_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "case": args.case, "model": args.model, "reference": args.reference,
        "confirmed": [sorted(p) for p in rec.proposed],
        "residual_a": rec.residual_a, "residual_b": rec.residual_b,
        "work": rec.work, "effort": rec.effort, "transcript": transcript,
    }, indent=2))

    print(f"\n=== {args.case} ({tag}) — {len(transcript)} turns, "
          f"{len(rec.proposed)} confirmed, residual a={len(rec.residual_a)} b={len(rec.residual_b)} ===\n")
    for i, turn in enumerate(transcript):
        side = turn["from"]
        q = turn.get("questions", []); a = turn.get("answers", [])
        p = turn.get("proposals", []); r = turn.get("ratifications", [])
        nc = turn.get("no_counterpart", [])
        print(f"[{i+1}] Agent {side}  done={turn.get('done')}")
        for x in q:
            print(f"      ASK  {x['their_id']}: {x['ask']}")
        for x in a:
            print(f"      ANSW {x['my_id']}: {x.get('gloss','')[:80]}")
        for x in p:
            print(f"      PROP {x['my_id']} <-> {x['their_id']}  conf={x['confidence']}  {x['rationale'][:70]}")
        for x in r:
            verdict = "ACCEPT" if x['accept'] else "REJECT"
            print(f"      RTFY {verdict} {x['their_id']}<->{x['my_id']}  {x['reason'][:60]}")
        if nc:
            print(f"      NO-COUNTERPART {nc}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("Read: are there ASK lines (agents interrogating) before they defer, or do they go "
          "straight to done? Many ASKs then honest deferral = a real finding; few ASKs = the "
          "protocol under-drives interrogation and should give agents more reason to probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
