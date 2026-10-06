#!/usr/bin/env python3
"""Multi-trial comparison on config_tapi_teas WITH the decisive experiment, for the cognition note.

Two arms, N trials each:
  * catalogue   -- Case 1: TwoAgentStackInstrumented (surfaces only; no standard named)
  * disclosure  -- Case 2: TwoAgentStackDisclosure (each side also advertises system/dialect)

Each trial reconciles, is scored against the validated key (precision, recall, which TRUE
correspondences were missed), and its transcript is written to results/trials/. A summary then
reports the recall distribution per arm and how often each true pair is missed -- the question
being whether Case 2's "reason past the decisive test" misses (t.cs<->i.tunnel, t.topo<->i.tetopo)
recur or were a one-draw fluke.

  python _shelf/orch/run_trials_tapi_exp.py --trials 3

Needs OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent            # study/_shelf/orch/
ROOT = HERE.parents[1]                             # -> study/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(HERE))

from reconcile.model import Case                                               # noqa: E402
from reconcile.schema_oracle import SchemaOracle                              # noqa: E402
from agent_twoagent_instrumented import TwoAgentStackInstrumented            # noqa: E402
from agent_twoagent_disclosure import TwoAgentStackDisclosure                # noqa: E402
from run import load_dotenv                                                   # noqa: E402


def score(confirmed, gold):
    corr = {frozenset((c["a"], c["b"])) for c in gold["correspondences"]}
    conf = {frozenset(p) for p in confirmed}
    tp = conf & corr
    prec = len(tp) / len(conf) if conf else 1.0
    rec = len(tp) / len(corr) if corr else 1.0
    missed = sorted(tuple(sorted(m)) for m in (corr - conf))
    return prec, rec, len(conf), missed, len(corr)


def run_arm(name, make_stack, case, gold, trials, outdir, model, budget, maxr):
    rows = []
    for t in range(trials):
        oracle = SchemaOracle(case, budget=budget)
        stack = make_stack(model, maxr, oracle)
        rec = stack.reconcile(case.model_a, case.model_b, reference=None,
                              placement="both_cognitive")
        tr = getattr(stack, "last_transcript", [])
        confirmed = [sorted(p) for p in rec.proposed]
        d = {"arm": name, "trial": t, "confirmed": confirmed,
             "residual_a": rec.residual_a, "residual_b": rec.residual_b,
             "work": rec.work, "transcript": tr}
        (outdir / f"{name}_t{t}.json").write_text(json.dumps(d, indent=2, default=str))
        prec, recl, nconf, missed, ncorr = score(confirmed, gold)
        rows.append({"t": t, "prec": prec, "recall": recl, "nconf": nconf,
                     "exp": rec.work.get("experiments"), "turns": rec.work.get("turns"),
                     "missed": missed})
        print(f"  [{name}] t{t}: confirmed={nconf}/{ncorr}  precision={prec:.2f}  recall={recl:.2f}  "
              f"experiments={rec.work.get('experiments')}  turns={rec.work.get('turns')}  "
              f"missed_true={missed}", flush=True)
    return rows


def summarise(name, rows):
    recalls = [r["recall"] for r in rows]
    precs = [r["prec"] for r in rows]
    exps = [r["exp"] for r in rows]
    miss_counter = Counter()
    for r in rows:
        for m in r["missed"]:
            miss_counter["<->".join(m)] += 1
    print(f"\n[{name}]  n={len(rows)}")
    print(f"  recall:   {['%.2f'%x for x in recalls]}   mean={sum(recalls)/len(recalls):.2f}")
    print(f"  precision:{['%.2f'%x for x in precs]}   mean={sum(precs)/len(precs):.2f}")
    print(f"  experiments: {exps}")
    if miss_counter:
        print(f"  missed true pairs (count/{len(rows)}): " +
              ", ".join(f"{k}={v}" for k, v in miss_counter.most_common()))
    else:
        print("  missed true pairs: none")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--case", default="config_tapi_teas")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--exp-budget", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--validate", action="store_true", help="offline plumbing check; no API call")
    args = ap.parse_args()

    load_dotenv()
    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    gold = json.loads((ROOT / "benchmark" / "cases" / args.case / "gold.json").read_text())
    outdir = ROOT / "results" / "trials"
    outdir.mkdir(parents=True, exist_ok=True)

    arms = [
        ("catalogue", lambda m, mr, o: TwoAgentStackInstrumented(model=m, max_rounds=mr, oracle=o)),
        ("disclosure", lambda m, mr, o: TwoAgentStackDisclosure(model=m, max_rounds=mr, oracle=o)),
    ]

    if args.validate:
        for name, mk in arms:
            s = mk(args.model, args.max_rounds, SchemaOracle(case, budget=args.exp_budget))
            print(f"[validate] {name}: stack={s.name}")
        print(f"[validate] case {args.case}: {len(case.model_a.by_id)} vs {len(case.model_b.by_id)} "
              f"concepts; {len(gold['correspondences'])} true correspondences. No API call made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    print(f"=== {args.trials} trials/arm on {args.case} + experiment (budget {args.exp_budget}, "
          f"max_rounds {args.max_rounds}), model {args.model} ===")
    allrows = {}
    for name, mk in arms:
        print(f"\n--- arm: {name} ---")
        allrows[name] = run_arm(name, mk, case, gold, args.trials, outdir,
                                args.model, args.exp_budget, args.max_rounds)

    print("\n==================== SUMMARY ====================")
    for name, _ in arms:
        summarise(name, allrows[name])
    print(f"\nPer-trial transcripts in {outdir}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
