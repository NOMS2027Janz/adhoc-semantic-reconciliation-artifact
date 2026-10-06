#!/usr/bin/env python3
"""Construct-then-bind runner (Track A / A1): the end-to-end standard-free protocol.

For each model and placement, the agents construct a thin shared reference from the two
models and then bind through it (``reconcile.construct``), with none pre-given. The row
is scored against the case gold exactly like the rest of the harness. With ``--baselines``
the no-reference and reference-given agent rows are recorded alongside, so the value of
*constructing the ground* sits between the reference-absent lower bound and the
reference-given upper bound in one table.

  # the headline: construct-then-bind on the standard-free case, across the ladder
  python pipeline/construct_then_bind.py --baselines \
      --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano \
      --placement both_cognitive,one_inert --trials 3

Deterministic parts are none — every row is a language-model run, so this needs an API
key (notes/SETUP_OPENAI.md). Rows stream to results/construct_then_bind_<case>.csv and
the run resumes over rows already present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile import Case                                              # noqa: E402
from reconcile.metrics import score                                    # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack             # noqa: E402
from reconcile.construct import construct_reference, bind_through_reference  # noqa: E402
from run import load_dotenv                                            # noqa: E402

CASES = ROOT / "benchmark" / "cases"
COLS = ["case", "condition", "model", "placement", "trial", "precision", "recall", "f1",
        "surviving_false_cognates", "residual", "constructed_entries",
        "construct_reasoning_tokens", "bind_reasoning_tokens"]


def _row(case, cond, model, placement, trial, rec, extra):
    s = score(rec, case.gold)
    return {"case": case.name, "condition": cond, "model": model, "placement": placement,
            "trial": trial, "precision": s.get("precision"), "recall": s.get("recall"),
            "f1": s.get("f1"), "surviving_false_cognates": s.get("surviving_false_cognates"),
            "residual": s.get("residual"),
            "constructed_entries": extra.get("entries", ""),
            "construct_reasoning_tokens": extra.get("construct_reasoning_tokens", ""),
            "bind_reasoning_tokens": extra.get("bind_reasoning_tokens", "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_cross_domain")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--placement", default="both_cognitive,one_inert")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--baselines", action="store_true",
                    help="also record the no-ref and reference-given agent rows")
    ap.add_argument("--save-refs", action="store_true",
                    help="save each constructed reference to results/construct_refs/ for inspection")
    ap.add_argument("--fresh", action="store_true", help="ignore existing rows (default: resume)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = Case.load(CASES / args.case)
    a, b = case.model_a, case.model_b
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    placements = [p.strip() for p in args.placement.split(",") if p.strip()]
    out = Path(args.out) if args.out else ROOT / "results" / f"construct_then_bind_{args.case}.csv"
    out.parent.mkdir(exist_ok=True)

    existing = []
    if out.exists() and not args.fresh:
        with out.open() as fh:
            existing = list(csv.DictReader(fh))
    done = {(r["condition"], r["model"], r["placement"], r["trial"]) for r in existing}
    fh = out.open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    for r in existing:
        w.writerow({k: r.get(k, "") for k in COLS})
    fh.flush()

    for model in models:
        for placement in placements:
            for t in range(args.trials):
                if args.baselines:
                    for use_ref in (False, True):
                        cond = "given-ref" if use_ref else "no-ref"
                        if (cond, model, placement, str(t)) in done:
                            continue
                        stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                        rec = stack.reconcile(a, b, reference=case.reference if use_ref else None,
                                              placement=placement)
                        w.writerow(_row(case, cond, model, placement, t, rec, {}))
                        fh.flush()
                        print(f"  {cond}: {model}/{placement}/t{t}", file=sys.stderr)

                if ("constructed", model, placement, str(t)) in done:
                    continue
                cref, e1 = construct_reference(a, b, model)
                if args.save_refs:
                    rdir = ROOT / "results" / "construct_refs"
                    rdir.mkdir(parents=True, exist_ok=True)
                    (rdir / f"{case.name}_{model}_{placement}_t{t}.json").write_text(
                        json.dumps({"model": model, "placement": placement, "trial": t,
                                    "n_entries": len(cref.entries),
                                    "entries": [asdict(e) for e in cref.entries]}, indent=1))
                bres, e2 = bind_through_reference(a, b, cref, model, placement)
                stack = OpenAIAgentStack(use_reference=True, model=model)
                rec = stack.to_reconciliation(bres, a, b, {**e1, **e2}, placement)
                w.writerow(_row(case, "constructed", model, placement, t, rec, {**e1, **e2}))
                fh.flush()
                print(f"  constructed: {model}/{placement}/t{t} "
                      f"({e1.get('entries')} entries)", file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
