#!/usr/bin/env python3
"""Dump the DISCLOSURE + instrumented two-agent transcript for ONE case.

Like ``two_agent_instrumented_transcript.py``, but uses ``TwoAgentStackDisclosure``: the
negotiation is preceded by the provenance-disclosure handshake (each side advertises its
system, dialect, and catalogue), which is recorded as the first two transcript entries, and
every negotiation turn still carries the ``plan`` channel.

  # clean negotiation, with the disclosure handshake
  python _shelf/orch/two_agent_disclosure_transcript.py

  # + the decisive virtual experiment
  python _shelf/orch/two_agent_disclosure_transcript.py --experiment

  # the cross-domain (bespoke-vs-bespoke) seam, with the experiment
  python _shelf/orch/two_agent_disclosure_transcript.py --case config_cross_domain --experiment

Writes results/transcript_disclosure_<case>_<mslug>_<ref|noref>[_exp].json. Needs OPENAI_API_KEY.
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
sys.path.insert(0, str(HERE))

from reconcile.model import Case                                               # noqa: E402
from agent_twoagent_disclosure import TwoAgentStackDisclosure                  # noqa: E402
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
                    help="offline checks only; no API call")
    args = ap.parse_args()

    load_dotenv()
    case = Case.load(ROOT / "benchmark" / "cases" / args.case)

    oracle = None
    if args.experiment:
        from reconcile.schema_oracle import SchemaOracle
        oracle = SchemaOracle(case, budget=args.exp_budget)

    stack = TwoAgentStackDisclosure(use_reference=args.reference, model=args.model,
                                    max_rounds=args.max_rounds, oracle=oracle)

    if args.validate:
        na, nb = len(case.model_a.by_id), len(case.model_b.by_id)
        print("[validate] OK -- no API call made.")
        print(f"  case {args.case}: A = {na} concepts ({case.model_a.system} / {case.model_a.dialect}), "
              f"B = {nb} concepts ({case.model_b.system} / {case.model_b.dialect})")
        print(f"  stack: {stack.name}  (experiment={'on' if oracle else 'off'}, "
              f"reference={args.reference}, max_rounds={args.max_rounds})")
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
    out = Path(args.out) if args.out else ROOT / "results" / f"transcript_disclosure_{args.case}_{mslug}_{tag}.json"
    payload = {
        "case": args.case,
        "model": args.model,
        "reference": args.reference,
        "experiment": args.experiment,
        "disclosure": True,
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
    print(f"wrote {out}  ({len(transcript)} transcript entries incl. 2 advertisements)")
    for e in transcript:
        if e.get("advertisement"):
            ad = e["advertisement"]
            print(f"\n[{e.get('from')}] ADVERTISES  system={ad.get('system')!r}  dialect={ad.get('dialect')!r}  "
                  f"({len(ad.get('catalogue', []))} concepts)")
            continue
        pl = e.get("plan", {}) or {}
        print(f"\n[{e.get('from')}] decision={pl.get('decision')!r} closeness={pl.get('closeness')}")
        if pl.get("need"):
            print(f"  need: {str(pl.get('need'))[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
