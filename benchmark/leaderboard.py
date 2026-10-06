#!/usr/bin/env python3
"""Leaderboard: score reconciliation stacks across the benchmark's schema cases.

The deterministic controls run offline; the language-model agent rows need an API key
(see ``notes/SETUP_OPENAI.md``). Rows are written to ``benchmark/leaderboard.csv`` so
stacks, models, and placements are comparable across cases in one table.

  # controls only, all schema cases (no API):
  python benchmark/leaderboard.py

  # add the language-model agent, no-reference and with-reference, across the ladder:
  python benchmark/leaderboard.py --agent \
      --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano \
      --placement both_cognitive,one_inert,both_inert --trials 3

The controls are always (re)written; --agent appends agent rows, resuming over any
already present so an interrupted run can be continued.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile import BaselineMatcher, Case, ReferenceReconciler, run_case   # noqa: E402
from reconcile.stacks.classical_matcher import ClassicalMatcher             # noqa: E402
from pack import classify, SETTING                                           # noqa: E402

COLS = ["case", "setting", "stack", "model", "placement", "uses_reference",
        "precision", "resolved_fraction", "f1", "surviving_false_cognates", "residual",
        "conf_correct_mean", "conf_wrong_mean", "calibration_gap", "brier",
        "confident_errors", "confident_error_rate",
        "scaling", "total_tokens", "reasoning_tokens", "latency_s"]

# The scoring code emits the metric under the internal key ``recall``; the published
# column is named ``resolved_fraction`` (the paper's and BENCHMARK.md's term for it).
_METRIC_ALIAS = {"resolved_fraction": "recall"}


def schema_cases() -> list[Path]:
    return [d for d in sorted((HERE / "cases").iterdir())
            if d.is_dir() and classify(d) == "schema"]


def _row(case_name: str, stack_label: str, model: str, r: dict) -> dict:
    out = {"case": case_name, "setting": SETTING.get(case_name, ""), "stack": stack_label,
           "model": model}
    for k in COLS:
        if k in ("case", "setting", "stack", "model"):
            continue
        out[k] = r.get(_METRIC_ALIAS.get(k, k), "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", action="store_true", help="also run the language-model agent (needs API)")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--placement", default="both_cognitive,one_inert,both_inert")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "leaderboard.csv"))
    args = ap.parse_args()

    cases = schema_cases()
    out = Path(args.out)
    fh = out.open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    n = 0
    errors = 0

    def emit(row):
        nonlocal n
        w.writerow(row)
        fh.flush()
        n += 1

    # 1. deterministic controls (offline), always fresh
    for d in cases:
        case = Case.load(d)
        emit(_row(case.name, "baseline (no reference)", "-",
                  run_case(case, BaselineMatcher(), placement="n/a")))
        emit(_row(case.name, "classical (lexical+structural)", "-",
                  run_case(case, ClassicalMatcher(), placement="n/a")))
        emit(_row(case.name, "reference control", "-",
                  run_case(case, ReferenceReconciler(), placement="n/a")))
        print(f"  controls: {case.name}", file=sys.stderr)

    # 2. optional language-model agent rows (streamed; a per-run error is skipped, not fatal)
    if args.agent:
        from run import load_dotenv
        load_dotenv()
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
            fh.close()
            return 2
        from reconcile.stacks.agent_openai import OpenAIAgentStack
        models = [m.strip() for m in args.model.split(",") if m.strip()]
        placements = [p.strip() for p in args.placement.split(",") if p.strip()]
        for d in cases:
            case = Case.load(d)
            for model in models:
                for placement in placements:
                    for use_ref in (False, True):
                        for t in range(args.trials):
                            label = f"agent ({'ref' if use_ref else 'no-ref'})"
                            try:
                                stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                                r = run_case(case, stack, placement=placement)
                            except Exception as e:  # noqa: BLE001 - one bad run must not sink the job
                                errors += 1
                                print(f"  ! skip {case.name}/{model}/{placement}/"
                                      f"{'ref' if use_ref else 'no-ref'}/t{t}: {e}", file=sys.stderr)
                                continue
                            emit(_row(case.name, label, model, r))
                            print(f"  agent: {case.name} / {model} / {placement} / "
                                  f"{'ref' if use_ref else 'no-ref'} / t{t}", file=sys.stderr)

    fh.close()
    print(f"\nwrote {out.name} — {n} row(s), {len(cases)} case(s), {errors} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
