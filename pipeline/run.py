#!/usr/bin/env python3
"""Run the walking skeleton: one case, with and without the shared reference.

    python run.py                      # runs config_tapi_teas, prints a table, writes results/
    python run.py --case config_tapi_teas
    python run.py --no-write           # print only

Adds src/ to the path so it runs with no install.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reconcile import BaselineMatcher, Case, ReferenceReconciler, run_case  # noqa: E402
from reconcile.metrics import METRIC_COLUMNS  # noqa: E402


def load_dotenv() -> None:
    """Minimal .env reader: set any KEY=VALUE lines that aren't already in the env.

    Lets you keep OPENAI_API_KEY in a local .env file (git-ignored) instead of
    exporting it each session. No dependency, and the real environment wins.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def format_table(rows: list[dict], show_tokens: bool = False) -> str:
    cols = ["stack", "placement", "uses_reference", "precision", "recall", "f1",
            "surviving_false_cognates", "residual", "scaling"]
    head = ["stack", "placement", "ref?", "prec", "recall", "f1", "false_cog", "resid", "scaling"]
    if show_tokens:
        cols += ["total_tokens", "reasoning_tokens", "latency_s"]
        head += ["tokens", "reason", "sec"]
    widths = [max(len(h), *(len(str(r.get(c, ""))) for r in rows)) for h, c in zip(head, cols)]
    def fmt(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    lines = [fmt(head), fmt(["-" * w for w in widths])]
    for r in rows:
        lines.append(fmt([r.get(c, "") for c in cols]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one reconciliation case, with and without the reference.")
    ap.add_argument("--case", default="config_tapi_teas")
    ap.add_argument("--placement", default="both_cognitive",
                    help="comma-separated placements for the agent: both_cognitive,one_inert")
    ap.add_argument("--model", default=None,
                    help="comma-separated model ids to sweep (default: OPENAI_MODEL or gpt-5.6)")
    ap.add_argument("--agent", action="store_true",
                    help="add the OpenAI agent stack (needs `pip install openai` and OPENAI_API_KEY)")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat each agent condition N times (LLM output varies run to run)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    case_dir = ROOT / "benchmark" / "cases" / args.case
    case = Case.load(case_dir)
    placements = [p.strip() for p in args.placement.split(",") if p.strip()]

    rows = [
        run_case(case, BaselineMatcher(), placement="n/a"),        # without reference (control)
        run_case(case, ReferenceReconciler(), placement="n/a"),    # with reference (control)
    ]

    if args.agent:
        load_dotenv()
        try:
            from reconcile.stacks.agent_openai import OpenAIAgentStack
        except ImportError:
            print("\n[agent] `openai` is not installed. Run: pip install openai", file=sys.stderr)
            return 2
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n[agent] OPENAI_API_KEY is not set. See SETUP_OPENAI.md.", file=sys.stderr)
            return 2
        models = [m.strip() for m in args.model.split(",")] if args.model else [None]
        models = [m for m in models if m] or [None]
        trials = max(1, args.trials)
        total = len(models) * len(placements) * 2 * trials
        i = 0
        for model in models:
            for placement in placements:
                for use_ref in (False, True):
                    stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                    cond = f"{stack.model} / {placement} / {'ref' if use_ref else 'no-ref'}"
                    for t in range(trials):
                        i += 1
                        print(f"[{i}/{total}] {cond}  trial {t + 1}/{trials} ... calling model",
                              file=sys.stderr, flush=True)
                        try:
                            row = run_case(case, stack, placement=placement)
                        except Exception as e:  # one bad call shouldn't lose the whole run
                            print(f"      ! error: {e}", file=sys.stderr, flush=True)
                            continue
                        print(f"      done: recall {row['recall']}, precision {row['precision']}, "
                              f"{row.get('total_tokens', '')} tokens, {row.get('latency_s', '')}s",
                              file=sys.stderr, flush=True)
                        rows.append(row)

    print(f"\nCase: {case.name}   placement: {args.placement}")
    print(f"  {case.model_a.system} ({case.model_a.dialect})  vs  "
          f"{case.model_b.system} ({case.model_b.dialect})")
    print(f"  gold: {len(case.gold.correspondences)} correspondences, "
          f"{len(case.gold.false_cognates)} planted false cognate(s)\n")
    print(format_table(rows, show_tokens=args.agent))

    base, ref = rows[0], rows[1]
    print("\nReading:")
    print(f"  Without the reference, the label pass reaches recall {base['recall']} and lets "
          f"{base['surviving_false_cognates']} false cognate through (precision {base['precision']}).")
    print(f"  With the thin reference, recall {ref['recall']} and precision {ref['precision']}, "
          f"the false cognate pre-empted, at {ref['bilateral_checks']} bilateral checks.")

    if not args.no_write:
        out = ROOT / "results" / f"{case.name}.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
