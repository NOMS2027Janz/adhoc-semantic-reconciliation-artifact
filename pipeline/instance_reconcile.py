#!/usr/bin/env python3
"""Driver for the instance-disambiguation study: cognition spectrum x evidence x
reference-variant x oracle-budget x model ladder, over the seeded hard instance case.

Staged, per the plan (each stage its own CSV, en-route row-by-row capture, resume):

  --stage smoke   one model, 3 placements, full evidence, no reference, unbounded oracle,
                  1 trial. Cheap shakeout of the loop and the case on real API.
  --stage 1       the spectrum + evidence headline: full 2^4 evidence factorial at
                  both_inert, a reduced evidence ladder at one_inert/both_cognitive;
                  models ladder; no reference; unbounded oracle; 6 trials.
  --stage 2       reference variants none/instance/invariant x placements x models,
                  full evidence, unbounded oracle, 6 trials.
  --stage 3       oracle-budget sweep {0,3,30} x {both_cognitive,one_inert} x models,
                  full evidence, no reference, 6 trials.

Correctness is the currency; interrogation/manipulation counts and tokens are effort.
A representative transcript per (placement) is saved for the write-up's demonstration.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.instance import load_instance_case                    # noqa: E402
from reconcile.stacks.agent_instance import InstanceAgentStack, EVIDENCE_ALL  # noqa: E402
from run import load_dotenv                                          # noqa: E402

PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
FULL_EVIDENCE = set(EVIDENCE_ALL)  # name,key,attrs,rels
REDUCED_LADDER = [
    {"name", "key", "attrs", "rels"},   # full
    {"name", "attrs", "rels"},          # no key
    {"key", "attrs", "rels"},           # no name (drop the trap surface)
    {"name", "key"},                    # lexical + key
    {"attrs", "rels"},                  # structure only (no lexical, no key)
    {"name"},                           # name only
]

COLUMNS = ["case", "model", "placement", "reference_variant", "evidence", "budget", "trial",
           "submitted", "proposed", "precision", "recall", "surviving_instance_fc",
           "experiment_only_recall", "residual_total", "residual_experiment_only",
           "residual_native", "confirmed_provision", "refuted_provision", "interrogated",
           "interrogate_calls", "provision_calls", "turns", "total_tokens", "reasoning_tokens",
           "latency_s", "mean_conf_correct", "mean_conf_incorrect", "surviving_traps"]


def ev_str(evidence: set) -> str:
    return "+".join(f for f in EVIDENCE_ALL if f in evidence) or "id-only"


def trap_names(case):
    return {frozenset((fc["a"], fc["b"])): f"{fc['a']}~{fc['b']}" for fc in case.gold.get("false_cognates", [])}


def score_row(case, rec, model, budget, trial, names, evidence) -> dict:
    proposed = set(rec.proposed)
    correct = case.correct_pairs
    fc = case.false_cognate_pairs
    eo = case.experiment_only_pairs
    tp = len(proposed & correct)
    fp = len(proposed - correct)
    fn = len(correct - proposed)
    a_only = {r["id"] for r in case.gold["residual"]["a_only"]}
    b_only = {r["id"] for r in case.gold["residual"]["b_only"]}
    matched = set().union(*proposed) if proposed else set()
    residual_native = len((a_only | b_only) - matched)  # native gaps correctly left residual

    eo_hit = proposed & eo
    conf_correct = [rec.confidence[p] for p in proposed & correct if p in rec.confidence]
    conf_incorrect = [rec.confidence[p] for p in proposed - correct if p in rec.confidence]
    surviving = ";".join(sorted(names.get(p, "?") for p in proposed & fc))
    return {
        "case": case.name, "model": model, "placement": rec.placement,
        "reference_variant": rec.reference_variant, "evidence": ev_str(evidence),
        "budget": "unbounded" if budget is None else budget, "trial": trial,
        "submitted": rec.submitted, "proposed": len(proposed),
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
        "surviving_instance_fc": len(proposed & fc),
        "experiment_only_recall": round(len(eo_hit) / len(eo), 3) if eo else "",
        "residual_total": len(rec.residual_a) + len(rec.residual_b),
        "residual_experiment_only": len(eo - proposed),
        "residual_native": residual_native,
        "confirmed_provision": len(rec.confirmed_pairs),
        "refuted_provision": "",  # filled from oracle log via transcript below
        "interrogated": len(rec.interrogated),
        "interrogate_calls": rec.oracle_calls.get("interrogate", 0),
        "provision_calls": rec.oracle_calls.get("provision", 0),
        "turns": rec.effort.get("turns", ""),
        "total_tokens": rec.effort.get("total_tokens", ""),
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "latency_s": rec.effort.get("latency_s", ""),
        "mean_conf_correct": round(sum(conf_correct) / len(conf_correct), 3) if conf_correct else "",
        "mean_conf_incorrect": round(sum(conf_incorrect) / len(conf_incorrect), 3) if conf_incorrect else "",
        "surviving_traps": surviving,
    }


def conditions_for_stage(stage: str, models: list[str]):
    """Yield (model, placement, reference_variant, evidence, budget) tuples."""
    if stage == "smoke":
        for m in models[:1]:
            for pl in PLACEMENTS:
                yield m, pl, "none", FULL_EVIDENCE, None
    elif stage == "1lite":
        # the reduced evidence ladder at all three placements — the shared cells the models
        # are compared on, without the full 2^4 factorial. For the weak model, tractable.
        for m in models:
            for pl in PLACEMENTS:
                for ev in REDUCED_LADDER:
                    yield m, pl, "none", set(ev), None
    elif stage == "1":
        for m in models:
            # full 2^4 evidence at both_inert
            for combo in itertools.product([False, True], repeat=len(EVIDENCE_ALL)):
                ev = {f for f, on in zip(EVIDENCE_ALL, combo) if on}
                yield m, "both_inert", "none", ev, None
            # reduced ladder at the live placements
            for pl in ("both_cognitive", "one_inert"):
                for ev in REDUCED_LADDER:
                    yield m, pl, "none", set(ev), None
    elif stage == "2":
        for m in models:
            for rv in ("none", "instance", "invariant"):
                for pl in PLACEMENTS:
                    yield m, pl, rv, FULL_EVIDENCE, None
    elif stage == "3":
        for m in models:
            for budget in (0, 3, None):
                for pl in ("both_cognitive", "one_inert"):
                    yield m, pl, "none", FULL_EVIDENCE, budget
    else:
        raise SystemExit(f"unknown stage '{stage}'")


def run_key(r):
    return (str(r["model"]), r["placement"], r["reference_variant"],
            str(r["evidence"]), str(r["budget"]), str(r["trial"]))


def cond_key(model, pl, rv, ev, budget, trial):
    return (str(model), pl, rv, ev_str(ev), "unbounded" if budget is None else str(budget), str(trial))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="smoke", help="smoke | 1 | 2 | 3")
    ap.add_argument("--case", default="instance_hard")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--budget", default=None,
                    help="override the oracle budget for all conditions: an int, or 'unbounded'")
    ap.add_argument("--inert-side", default="b")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (default: resume)")
    ap.add_argument("--save-transcripts", action="store_true",
                    help="save a representative transcript per placement (trial 0, full evidence)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = load_instance_case(ROOT / "benchmark" / "cases" / args.case)
    names = trap_names(case)
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = 1 if args.stage == "smoke" else max(1, args.trials)
    plan = list(conditions_for_stage(args.stage, models))
    total = len(plan) * trials

    # one CSV per stage; when a SINGLE model is passed, suffix it with the model so that
    # per-model terminals can run in parallel without clashing on one file.
    suffix = f"_{models[0]}" if len(models) == 1 else ""
    out = ROOT / "results" / f"instance_{args.case}_stage{args.stage}{suffix}.csv"
    tdir = ROOT / "results" / "instance_transcripts"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)
        if args.save_transcripts:
            tdir.mkdir(exist_ok=True)

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    fh = writer = None
    if write:
        fresh_file = args.fresh or not out.exists()
        fh = out.open("w" if fresh_file else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh_file:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    print(f"Instance study stage {args.stage}: {len(plan)} conditions x {trials} trials "
          f"= {total} runs, case {args.case}", file=sys.stderr)

    budget_override_set = args.budget is not None
    budget_override = None if args.budget in (None, "unbounded") else int(args.budget)

    i = 0
    seen_transcript = set()
    for (model, pl, rv, ev, budget) in plan:
        if budget_override_set:
            budget = budget_override
        for t in range(trials):
            i += 1
            key = cond_key(model, pl, rv, ev, budget, t)
            if key in done:
                print(f"[{i}/{total}] {key}  (captured, skip)", file=sys.stderr, flush=True)
                continue
            print(f"[{i}/{total}] {model}/{pl}/{rv}/{ev_str(ev)}/budget={budget}/trial {t+1}",
                  file=sys.stderr, flush=True)
            stack = InstanceAgentStack(case, model=model, reference_variant=rv,
                                       evidence=ev, inert_side=args.inert_side, budget=budget)
            try:
                rec = stack.reconcile(placement=pl)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            r = score_row(case, rec, model, budget, t, names, ev)
            r["refuted_provision"] = sum(1 for s in rec.transcript
                                         if s.get("step") == "virtual_provision"
                                         and s["result"]["answer"].get("confirmed") is False)
            rows.append(r); done.add(key)
            if writer is not None:
                writer.writerow(r); fh.flush()
            if (write and args.save_transcripts and t == 0 and ev == FULL_EVIDENCE
                    and rv == "none" and (model, pl) not in seen_transcript):
                seen_transcript.add((model, pl))
                (tdir / f"{args.case}_{model}_{pl}.json").write_text(
                    json.dumps({"model": model, "placement": pl, "row": r,
                                "transcript": rec.transcript}, indent=2))
            print(f"      prec {r['precision']} rec {r['recall']} eo-rec {r['experiment_only_recall']} "
                  f"fc {r['surviving_instance_fc']} resid {r['residual_total']} "
                  f"(eo {r['residual_experiment_only']}) probes[i{r['interrogate_calls']}/p{r['provision_calls']}] "
                  f"traps[{r['surviving_traps']}]", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarize(rows):
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

    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    for m in models:
        print(f"\n{'#'*10} model: {m} {'#'*10}")
        for pl in PLACEMENTS:
            sel = [r for r in rows if r["model"] == m and r["placement"] == pl]
            if not sel:
                continue
            print(f"  {pl:15s} prec {mean(sel,'precision'):.2f}  rec {mean(sel,'recall'):.2f}  "
                  f"eo-rec {mean(sel,'experiment_only_recall'):.2f}  "
                  f"fc {mean(sel,'surviving_instance_fc'):.2f}  "
                  f"resid {mean(sel,'residual_total'):.1f} (eo {mean(sel,'residual_experiment_only'):.1f})  "
                  f"probes i{mean(sel,'interrogate_calls'):.1f}/p{mean(sel,'provision_calls'):.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
