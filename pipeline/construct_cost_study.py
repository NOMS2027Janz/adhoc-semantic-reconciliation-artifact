#!/usr/bin/env python3
"""Construct-then-bind COST-BENEFIT study (Track A / A6): where is constructing the
reference load-bearing, and where is it redundant expenditure of cognition?

Construct-then-bind (A1) showed that in the standard-free setting a capable agent can
BUILD the shared reference itself and then bind through it (0.40 -> 0.93). The natural
follow-on question: if that works where no standard exists, is it worth doing everywhere?
Constructing a reference COSTS cognition; that cost is only justified by the *marginal*
improvement it buys over what you would have got without it. Where an effective reference
already exists (a public standard) or a cheap shortcut is at hand (rich mutual evidence),
the construction spend is largely redundant.

This runner measures, for each schema case, THREE conditions at each placement and model,
recording both the CLOSE (precision, resolved fraction, surviving false cognates)
and the COGNITION SPENT (reasoning tokens), with the construct step and the bind step
instrumented SEPARATELY:

  * no-ref      bind with no reference at all         (lower bound; agent reasons unaided)
  * constructed build the reference, then bind        (construct spend + bind spend)
  * given-ref   bind through the case's given reference(upper bound; a standard pre-exists)

Read it across the settings: setting 1 (config_big_hard) and setting 4
(config_observability) already carry an effective reference, so we expect constructed's
close to MATCH given-ref while construction spends cognition that bought little -> the
redundant regime. Setting 3 (config_cross_domain) has no standard, so we expect
constructed's close to sit far above no-ref -> construction is load-bearing. Placement is
the shortcut axis at schema level: both-cognitive = full mutual evidence (a cheap shortcut
to the ground); one-inert = one side mute (shortcut degraded).

  # the whole study, three primary schema settings, across the ladder (needs an API key)
  python pipeline/construct_cost_study.py \
      --case config_big_hard,config_cross_domain,config_observability \
      --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano \
      --placement both_cognitive,one_inert --trials 3

Rows stream to results/construct_cost_<case>.csv (one file per case) and the run resumes
over rows already present. Offline test: tests/test_construct_cost_study.py (no API/network).
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

from reconcile import Case                                                     # noqa: E402
from reconcile.metrics import score                                           # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack                    # noqa: E402
from reconcile.construct import construct_reference, bind_through_reference   # noqa: E402
from run import load_dotenv                                                   # noqa: E402

CASES = ROOT / "benchmark" / "cases"
CONDITIONS = ("no-ref", "constructed", "given-ref")
COLS = ["case", "condition", "model", "placement", "trial",
        "precision", "recall", "surviving_false_cognates", "residual",
        "constructed_entries",
        "construct_reasoning_tokens", "construct_total_tokens",
        "bind_reasoning_tokens", "bind_total_tokens",
        "total_reasoning_tokens", "total_tokens"]


def _num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _base_row(case, cond, model, placement, trial, rec):
    """no-ref / given-ref: one agent binding call; its effort IS the bind spend."""
    s = score(rec, case.gold)
    eff = getattr(rec, "effort", {}) or {}
    br, bt = eff.get("reasoning_tokens"), eff.get("total_tokens")
    return {"case": case.name, "condition": cond, "model": model, "placement": placement,
            "trial": trial, "precision": s.get("precision"), "recall": s.get("recall"),
            "surviving_false_cognates": s.get("surviving_false_cognates"),
            "residual": s.get("residual"), "constructed_entries": "",
            "construct_reasoning_tokens": 0, "construct_total_tokens": 0,
            "bind_reasoning_tokens": br, "bind_total_tokens": bt,
            "total_reasoning_tokens": _num(br), "total_tokens": _num(bt)}


def _constructed_row(case, model, placement, trial, rec, e1, e2):
    """constructed: construct spend + bind spend, summed into the totals."""
    s = score(rec, case.gold)
    cr, ct = e1.get("construct_reasoning_tokens"), e1.get("construct_total_tokens")
    br, bt = e2.get("bind_reasoning_tokens"), e2.get("bind_total_tokens")
    return {"case": case.name, "condition": "constructed", "model": model,
            "placement": placement, "trial": trial,
            "precision": s.get("precision"), "recall": s.get("recall"),
            "surviving_false_cognates": s.get("surviving_false_cognates"),
            "residual": s.get("residual"), "constructed_entries": e1.get("entries", ""),
            "construct_reasoning_tokens": cr, "construct_total_tokens": ct,
            "bind_reasoning_tokens": br, "bind_total_tokens": bt,
            "total_reasoning_tokens": _num(cr) + _num(br),
            "total_tokens": _num(ct) + _num(bt)}


def _run_case(case_name, models, placements, trials, fresh, out_dir):
    case = Case.load(CASES / case_name)
    a, b = case.model_a, case.model_b
    out = out_dir / f"construct_cost_{case_name}.csv"
    out.parent.mkdir(exist_ok=True)

    existing = []
    if out.exists() and not fresh:
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
            for t in range(trials):
                # no-ref and given-ref: a single binding call each
                for cond, use_ref in (("no-ref", False), ("given-ref", True)):
                    if (cond, model, placement, str(t)) in done:
                        continue
                    stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                    rec = stack.reconcile(a, b, reference=case.reference if use_ref else None,
                                          placement=placement)
                    w.writerow(_base_row(case, cond, model, placement, t, rec))
                    fh.flush()
                    print(f"  {case_name} {cond}: {model}/{placement}/t{t}", file=sys.stderr)
                # constructed: build the reference, then bind through it
                if ("constructed", model, placement, str(t)) in done:
                    continue
                cref, e1 = construct_reference(a, b, model)
                bres, e2 = bind_through_reference(a, b, cref, model, placement)
                stack = OpenAIAgentStack(use_reference=True, model=model)
                rec = stack.to_reconciliation(bres, a, b, {**e1, **e2}, placement)
                w.writerow(_constructed_row(case, model, placement, t, rec, e1, e2))
                fh.flush()
                print(f"  {case_name} constructed: {model}/{placement}/t{t} "
                      f"({e1.get('entries')} entries)", file=sys.stderr)
    fh.close()
    return out


def _summarise(out_paths):
    """Print a per-case, per-condition table: mean close and mean cognition spent."""
    import statistics as st
    print("\n" + "=" * 78)
    print("SUMMARY  (mean over models x placements x trials; spend = reasoning tokens)")
    print("=" * 78)
    hdr = f"{'case':22} {'condition':12} {'prec':>5} {'resfrac':>7} {'survFC':>6} {'spend':>8}"
    for out in out_paths:
        if not out.exists():
            continue
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        print("\n" + hdr)
        print("-" * len(hdr))
        for cond in CONDITIONS:
            rc = [r for r in rows if r["condition"] == cond]
            if not rc:
                continue
            def m(key):
                vals = [float(r[key]) for r in rc if r.get(key) not in ("", None)]
                return st.mean(vals) if vals else float("nan")
            print(f"{rc[0]['case']:22} {cond:12} {m('precision'):5.2f} {m('recall'):7.2f} "
                  f"{m('surviving_false_cognates'):6.2f} {m('total_reasoning_tokens'):8.0f}")
    print("\nReading it: where 'constructed' resolved fraction ~= 'given-ref' AND 'no-ref' resolved")
    print("fraction is already high, the construct spend bought little -> redundant. Where 'constructed'")
    print("resolved fraction sits far above 'no-ref', construction is load-bearing.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_big_hard,config_cross_domain,config_observability")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--placement", default="both_cognitive,one_inert")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="ignore existing rows (default: resume)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    cases = [c.strip() for c in args.case.split(",") if c.strip()]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    placements = [p.strip() for p in args.placement.split(",") if p.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results"

    out_paths = []
    for case_name in cases:
        print(f"\n=== {case_name} ===", file=sys.stderr)
        out_paths.append(_run_case(case_name, models, placements, args.trials, args.fresh, out_dir))
    _summarise(out_paths)
    print("wrote " + ", ".join(p.name for p in out_paths), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
