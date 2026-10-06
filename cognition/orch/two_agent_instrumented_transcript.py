#!/usr/bin/env python3
"""Dump the INSTRUMENTED two-agent transcript (with the per-turn planning voice) for ONE case.

Like ``two_agent_transcript.py``, but uses ``TwoAgentStackInstrumented`` so every turn also
carries a ``plan`` object -- the agent's orchestration meta-reasoning -- for the annotated,
commented traces in the orchestration paper.

  # the clean negotiation (scope -> hypothesise -> winnow), strong agent
  python pipeline/two_agent_instrumented_transcript.py

  # + the decisive virtual experiment (tool-invoked verification), strong agent
  python pipeline/two_agent_instrumented_transcript.py --experiment

  # the standard-free seam, with the experiment
  python pipeline/two_agent_instrumented_transcript.py --case config_cross_domain --experiment

  # the SAME case at a weak agent, for the capability-banding contrast
  python pipeline/two_agent_instrumented_transcript.py --model gpt-5-nano

Writes results/transcript_instrumented_<case>_<ref|noref>[_exp].json and prints a short
per-turn planning summary. Needs OPENAI_API_KEY (see notes/SETUP_OPENAI.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # study/_shelf/orch/
ROOT = HERE.parents[1]                             # -> study/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(HERE))                      # this dir, for the instrumented stack module

from reconcile.model import Case                                               # noqa: E402
from agent_twoagent_instrumented import TwoAgentStackInstrumented              # noqa: E402
from run import load_dotenv                                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_tapi_teas")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--reference", action="store_true", help="run WITH the shared reference")
    ap.add_argument("--experiment", action="store_true",
                    help="enable the decisive virtual experiment (SchemaOracle)")
    ap.add_argument("--exp-budget", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--validate", action="store_true",
                    help="offline checks only; no API call (per study convention)")
    args = ap.parse_args()

    load_dotenv()

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)

    oracle = None
    if args.experiment:
        from reconcile.schema_oracle import SchemaOracle
        oracle = SchemaOracle(case, budget=args.exp_budget)

    stack = TwoAgentStackInstrumented(use_reference=args.reference, model=args.model,
                                      max_rounds=args.max_rounds, oracle=oracle)

    if args.validate:
        from agent_twoagent_instrumented import TurnI
        na, nb = len(case.model_a.by_id), len(case.model_b.by_id)
        schema = TurnI.model_json_schema()
        print("[validate] OK -- no API call made.")
        print(f"  case {args.case}: model A = {na} concepts, model B = {nb} concepts")
        print(f"  stack: {stack.name}  (experiment={'on' if oracle else 'off'}, "
              f"reference={args.reference}, max_rounds={args.max_rounds})")
        print(f"  turn schema fields: {list(schema['properties'])}")
        print(f"  plan fields: {list(schema['$defs']['Plan']['properties'])}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    reference = case.reference if args.reference else None
    rec = stack.reconcile(case.model_a, case.model_b, reference=reference,
                          placement="both_cognitive")
    transcript = getattr(stack, "last_transcript", [])

    tag = ("ref" if args.reference else "noref") + ("_exp" if args.experiment else "")
    mslug = args.model.replace("/", "-").replace(".", "")
    out = Path(args.out) if args.out else ROOT / "results" / f"transcript_instrumented_{args.case}_{mslug}_{tag}.json"
    payload = {
        "case": args.case,
        "model": args.model,
        "reference": args.reference,
        "experiment": args.experiment,
        "confirmed": [sorted(p) for p in rec.proposed],
        "residual_a": rec.residual_a,
        "residual_b": rec.residual_b,
        "work": rec.work,
        "effort": rec.effort,
        "transcript": transcript,
    }
    if oracle is not None:
        payload["experiment_log"] = [
            {"a": r.a_id, "b": r.b_id, "ok": r.ok, "confirmed": r.confirmed, "relation": r.relation}
            for r in oracle.log]

    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}  ({len(transcript)} turns)")
    for e in transcript:
        pl = e.get("plan", {}) or {}
        print(f"\n[{e.get('from')}] decision={pl.get('decision')!r} "
              f"closeness={pl.get('closeness')}")
        if pl.get("need"):
            print(f"  need: {str(pl.get('need'))[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
