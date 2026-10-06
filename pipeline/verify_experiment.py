#!/usr/bin/env python3
"""Reconcile -> verify -> repair, and count the errors that survive verification.

    python verify_experiment.py --case config_big_hard --trials 3 \
      --placement both_cognitive,one_inert,both_inert --model gpt-5.6-sol,gpt-5-nano

Tests the correctness form of H2: with the framework's verification step actually
run, do the errors that slip through the proposal get caught — and does that
catching weaken as cognition recedes, leaving the reference's error-prevention to
matter most where verification is weakest? Verification reasons from the invariants
and structure only; it never sees the gold. Writes results/verify_<case>.csv with a
stage column (pre = proposed, post = after verify-and-repair).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Case  # noqa: E402
from reconcile.metrics import score  # noqa: E402
from run import load_dotenv  # noqa: E402

COLUMNS = ["case", "model", "placement", "uses_reference", "trial", "stage",
           "precision", "recall", "surviving_false_cognates", "residual",
           "reasoning_tokens", "verify_reasoning_tokens"]


def _row(case, model, placement, use_ref, trial, stage, rec, gold):
    m = score(rec, gold)
    return {
        "case": case, "model": model, "placement": placement,
        "uses_reference": use_ref, "trial": trial, "stage": stage,
        "precision": m["precision"], "recall": m["recall"],
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "verify_reasoning_tokens": rec.effort.get("verify_reasoning_tokens", ""),
    }


def summarize(rows):
    placements = []
    for r in rows:
        if r["placement"] not in placements:
            placements.append(r["placement"])
    def mean(rs, k):
        xs = [float(x[k]) for x in rs if x[k] not in ("", None)]
        return sum(xs) / len(xs) if xs else 0.0
    print("\n=== Surviving false cognates and precision, proposal (pre) vs after verify (post) ===")
    for pl in placements:
        print(f"\n  placement: {pl}")
        for ref in ("False", "True"):
            tag = "with ref   " if ref == "True" else "without ref"
            pre = [r for r in rows if r["placement"] == pl and r["uses_reference"] == ref and r["stage"] == "pre"]
            post = [r for r in rows if r["placement"] == pl and r["uses_reference"] == ref and r["stage"] == "post"]
            if not pre:
                continue
            print(f"    {tag}:  false cognates {mean(pre,'surviving_false_cognates'):.2f} -> "
                  f"{mean(post,'surviving_false_cognates'):.2f}   |   "
                  f"precision {mean(pre,'precision'):.2f} -> {mean(post,'precision'):.2f}")
    print("\n=== Errors surviving verification (post), by placement x reference ===")
    print("    (does the safety net catch less as cognition recedes?)")
    hdr = "    " + "reference".ljust(14) + "".join(pl.ljust(18) for pl in placements)
    print(hdr)
    for ref, tag in (("False", "without ref"), ("True", "with ref")):
        cells = []
        for pl in placements:
            post = [r for r in rows if r["placement"] == pl and r["uses_reference"] == ref and r["stage"] == "post"]
            cells.append(f"{mean(post,'surviving_false_cognates'):.2f}".ljust(18))
        print("    " + tag.ljust(14) + "".join(cells))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="config_big_hard")
    ap.add_argument("--placement", default="both_cognitive,one_inert,both_inert")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    placements = [p.strip() for p in args.placement.split(",") if p.strip()]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = max(1, args.trials)
    total = len(models) * len(placements) * 2 * trials
    i = 0
    rows = []
    for model in models:
        for placement in placements:
            for use_ref in (False, True):
                stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                cond = f"{model} / {placement} / {'ref' if use_ref else 'no-ref'}"
                for t in range(trials):
                    i += 1
                    print(f"[{i}/{total}] {cond}  trial {t+1}/{trials} ... reconcile+verify",
                          file=sys.stderr, flush=True)
                    try:
                        pre, post = stack.reconcile_and_verify(
                            case.model_a, case.model_b,
                            reference=(case.reference if use_ref else None), placement=placement)
                    except Exception as e:
                        print(f"      ! error: {e}", file=sys.stderr, flush=True)
                        continue
                    rp = _row(args.case, str(model), placement, str(use_ref), t, "pre", pre, case.gold)
                    rq = _row(args.case, str(model), placement, str(use_ref), t, "post", post, case.gold)
                    rows.extend([rp, rq])
                    print(f"      false cognates {rp['surviving_false_cognates']} -> "
                          f"{rq['surviving_false_cognates']}, precision {rp['precision']} -> {rq['precision']}",
                          file=sys.stderr, flush=True)

    summarize(rows)
    if not args.no_write and rows:
        out = ROOT / "results" / f"verify_{args.case}.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
