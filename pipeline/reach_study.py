#!/usr/bin/env python3
"""The reach study: how far an agent's reconciliation reaches as a function of what it can
SEE and what it can ASK - its interrogations - with cognitive power and liveness held at the top.

Two axes, over the seeded hard instance case, at both_cognitive with the strong model unless
told otherwise:

  visibility  (what the agent can see about each individual: the evidence mask)
      key            id + opaque shared key only
      key+name       + local names (introduces the shared-name trap)
      key+name+attrs + attributes
      key+name+attrs+rels   + topology (full)

  interrogation  (what the agent may DO on a live side)
      none        no probing; decide from the visible records
      lookup      interrogate a NAMED attribute only (no discovery, no experiment)
      discovery   interrogate may omit the attribute -> the side lists its facts
      full        discovery + virtual_provision (the decisive virtual experiment)

Modes:
  --mode grid    visibility x interrogation at one model (default sol) - the headline: reach
                 climbs from a floor to full as either axis opens, and discovery is the hinge.
  --mode ladder  interrogation x model at full visibility - reach x power: the ceiling is set by
                 the model, the shape by the interrogation level.
  --mode both    run both grids.

Budget is unbounded throughout: the reach axis is about the KIND of access, not the amount
(the amount is the separate oracle-budget sweep in instance_reconcile.py --stage 3).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.instance import load_instance_case                       # noqa: E402
from reconcile.stacks.agent_instance import (                           # noqa: E402
    InstanceAgentStack, EVIDENCE_ALL, INTERROGATION_LEVELS)
from run import load_dotenv                                             # noqa: E402
from instance_reconcile import score_row, trap_names, ev_str           # noqa: E402

# visibility ladder: cumulative evidence, from the opaque key alone up to the full record.
VISIBILITY = [
    {"key"},
    {"key", "name"},
    {"key", "name", "attrs"},
    {"key", "name", "attrs", "rels"},
]
FULL_VIS = {"key", "name", "attrs", "rels"}
LADDER = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]

COLUMNS = ["case", "model", "placement", "interrogation", "evidence", "reference_variant",
           "budget", "trial", "submitted", "proposed", "precision", "recall",
           "surviving_instance_fc", "experiment_only_recall", "residual_total",
           "residual_experiment_only", "residual_native", "confirmed_provision",
           "refuted_provision", "interrogated", "interrogate_calls", "provision_calls",
           "turns", "total_tokens", "reasoning_tokens", "latency_s", "mean_conf_correct",
           "mean_conf_incorrect", "surviving_traps"]


def conditions(mode: str, models: list[str]):
    """Yield (model, evidence, interrogation) cells."""
    if mode in ("grid", "both"):
        for ev in VISIBILITY:
            for aff in INTERROGATION_LEVELS:
                yield models[0], set(ev), aff
    if mode in ("ladder", "both"):
        for m in models:
            for aff in INTERROGATION_LEVELS:
                yield m, set(FULL_VIS), aff


def cell_key(model, ev, aff, trial):
    return (str(model), ev_str(ev), aff, str(trial))


def row_key(r):
    return (str(r["model"]), str(r["evidence"]), str(r["interrogation"]), str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="grid", help="grid | ladder | both")
    ap.add_argument("--case", default="instance_hard")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano",
                    help="comma list; grid uses the first, ladder uses all")
    ap.add_argument("--placement", default="both_cognitive")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--budget", default="unbounded", help="oracle budget (int or 'unbounded')")
    ap.add_argument("--inert-side", default="b")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (default: resume)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = load_instance_case(ROOT / "benchmark" / "cases" / args.case)
    names = trap_names(case)
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    budget = None if args.budget in (None, "unbounded") else int(args.budget)
    trials = max(1, args.trials)
    plan = list(conditions(args.mode, models))
    total = len(plan) * trials

    out = ROOT / "results" / f"reach_{args.case}_{args.mode}.csv"
    write = not args.no_write
    rows, done = [], set()
    if write:
        out.parent.mkdir(exist_ok=True)
        if out.exists() and not args.fresh:
            with out.open(newline="") as f:
                rows = list(csv.DictReader(f))
            done = {row_key(r) for r in rows}
            print(f"Resuming: {len(done)} cells already captured in {out.name}", file=sys.stderr)

    fh = writer = None
    if write:
        fresh_file = args.fresh or not out.exists()
        fh = out.open("w" if fresh_file else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh_file:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    print(f"Reach study [{args.mode}]: {len(plan)} cells x {trials} trials = {total} runs, "
          f"case {args.case}, placement {args.placement}, budget {args.budget}", file=sys.stderr)

    i = 0
    for (model, ev, aff) in plan:
        for t in range(trials):
            i += 1
            key = cell_key(model, ev, aff, t)
            if key in done:
                print(f"[{i}/{total}] {key}  (captured, skip)", file=sys.stderr, flush=True)
                continue
            print(f"[{i}/{total}] {model} / vis={ev_str(ev)} / ask={aff} / trial {t+1}",
                  file=sys.stderr, flush=True)
            stack = InstanceAgentStack(case, model=model, reference_variant="none",
                                       evidence=ev, inert_side=args.inert_side,
                                       budget=budget, interrogation=aff)
            try:
                rec = stack.reconcile(placement=args.placement)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            r = score_row(case, rec, model, budget, t, names, ev)
            r["interrogation"] = aff
            r["refuted_provision"] = sum(1 for s in rec.transcript
                                         if s.get("step") == "virtual_provision"
                                         and s["result"]["answer"].get("confirmed") is False)
            rows.append(r); done.add(key)
            if writer is not None:
                writer.writerow({c: r.get(c, "") for c in COLUMNS}); fh.flush()
            print(f"      prec {r['precision']} rec {r['recall']} "
                  f"eo-rec {r['experiment_only_recall']} fc {r['surviving_instance_fc']} "
                  f"resid {r['residual_total']} (eo {r['residual_experiment_only']}) "
                  f"probes[i{r['interrogate_calls']}/p{r['provision_calls']}]",
                  file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows, args.mode)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarize(rows, mode):
    if not rows:
        return

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, k):
        xs = [fl(r[k]) for r in rs if fl(r[k]) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n{'#'*12} reach [{mode}]: mean experiment-only resolved fraction {'#'*12}")
    if mode in ("grid", "both"):
        m0 = rows[0]["model"]
        vis_labels = [ev_str(v) for v in VISIBILITY]
        print(f"\n  grid (model {m0}) - rows: visibility, cols: interrogation")
        header = "    " + f"{'visibility':22s}" + "".join(f"{a:>11s}" for a in INTERROGATION_LEVELS)
        print(header)
        for vl in vis_labels:
            cells = []
            for a in INTERROGATION_LEVELS:
                sel = [r for r in rows if r["model"] == m0 and r["evidence"] == vl
                       and r["interrogation"] == a]
                cells.append(f"{mean(sel,'experiment_only_recall'):>11.2f}" if sel else f"{'-':>11s}")
            print(f"    {vl:22s}" + "".join(cells))
    if mode in ("ladder", "both"):
        print(f"\n  ladder (full visibility) - rows: model, cols: interrogation")
        header = "    " + f"{'model':22s}" + "".join(f"{a:>11s}" for a in INTERROGATION_LEVELS)
        print(header)
        for m in LADDER:
            cells = []
            for a in INTERROGATION_LEVELS:
                sel = [r for r in rows if r["model"] == m and r["evidence"] == ev_str(FULL_VIS)
                       and r["interrogation"] == a]
                cells.append(f"{mean(sel,'experiment_only_recall'):>11.2f}" if sel else f"{'-':>11s}")
            print(f"    {m:22s}" + "".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
