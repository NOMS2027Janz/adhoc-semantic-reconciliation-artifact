#!/usr/bin/env python3
"""Single-command staged runner for the observability setting (setting 4).

    python observability_study.py --stage all                 # launch-and-leave, all phases
    python observability_study.py --stage all --model gpt-5-mini   # one model per terminal

Four phases, each its own segmented CSV, a live phase banner, checkpointed and resumable:

  Phase 1/4  Act 1 — schema binding      (legacy vs NMOP; RFC 9940 reference; alarm/anomaly cognate;
                                          the alarm decomposition) — spectrum x reference x model
  Phase 2/4  Act 1 — alarm/anomaly co-reference (which alarm and which anomaly are one condition)
  Phase 3/4  Act 2 — the pragmatic verdict (act/watch/suppress), pragmatics ON vs OFF x context
  Phase 4/4  Act 2 — multi-symptom cross-layer correlation into incidents, pragmatics ON vs OFF

Correctness is the currency. Phases 1-2 reuse the schema and instance harnesses across the
cognition spectrum; phases 3-4 turn the semantics+pragmatics ON and OFF (the demo's toggle) to
measure what the pragmatics carry.
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
from reconcile.model import Case                                            # noqa: E402
from reconcile.metrics import score as schema_score                        # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack                  # noqa: E402
from reconcile.instance import load_instance_case                          # noqa: E402
from reconcile.stacks.agent_instance import InstanceAgentStack, EVIDENCE_ALL  # noqa: E402
from reconcile.observability import load_act2                              # noqa: E402
from reconcile.stacks.agent_observability import VerdictStack, CorrelationStack  # noqa: E402
from run import load_dotenv                                                # noqa: E402

SPECTRUM = ["both_cognitive", "one_inert", "both_inert"]
PHASE_LABEL = {1: "Act 1 · schema binding", 2: "Act 1 · alarm/anomaly co-reference",
               3: "Act 2 · pragmatic verdict (act/watch/suppress)",
               4: "Act 2 · multi-symptom correlation"}
CASE = "config_observability"


def banner(p, i, total, detail):
    print(f"=== Phase {p}/4 · {PHASE_LABEL[p]} ===   [{i:>3}/{total}]  {detail}",
          file=sys.stderr, flush=True)


def suffix(models):
    return f"_{models[0]}" if len(models) == 1 else ""


def open_csv(path, cols, fresh):
    rows = []
    if path.exists() and not fresh:
        rows = list(csv.DictReader(path.open(newline="")))
    fresh_file = fresh or not path.exists()
    fh = path.open("w" if fresh_file else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if fresh_file:
        w.writeheader(); fh.flush(); rows = []
    return fh, w, rows


# ---------------------------------------------------------------------------------------
# Phase 1: Act 1 schema binding
# ---------------------------------------------------------------------------------------
P1_COLS = ["case", "phase", "model", "placement", "reference", "trial", "precision", "recall",
           "f1", "surviving_false_cognates", "residual", "total_tokens", "reasoning_tokens", "latency_s"]


def phase1(case, models, trials, fresh, write):
    path = ROOT / "results" / f"obs_{CASE}_phase1{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P1_COLS, fresh) if write else (None, None, []))
    done = {(r["model"], r["placement"], r["reference"], r["trial"]) for r in rows}
    plan = [(m, pl, ref) for m in models for pl in SPECTRUM for ref in (False, True)]
    total = len(plan) * trials
    i = 0
    for (m, pl, ref) in plan:
        for t in range(trials):
            i += 1
            key = (m, pl, "ref" if ref else "no-ref", str(t))
            if key in done:
                banner(1, i, total, f"{m}/{pl}/{'ref' if ref else 'no-ref'}/t{t} (skip)"); continue
            banner(1, i, total, f"{m}/{pl}/{'ref' if ref else 'no-ref'}/trial {t+1}")
            stack = OpenAIAgentStack(use_reference=ref, model=m)
            try:
                rec = stack.reconcile(case.model_a, case.model_b,
                                      reference=case.reference if ref else None, placement=pl)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True); continue
            s = schema_score(rec, case.gold)
            row = {"case": CASE, "phase": "schema", "model": m, "placement": pl,
                   "reference": "ref" if ref else "no-ref", "trial": t,
                   "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
                   "surviving_false_cognates": s["surviving_false_cognates"], "residual": s["residual"],
                   "total_tokens": s["total_tokens"], "reasoning_tokens": s["reasoning_tokens"],
                   "latency_s": s["latency_s"]}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      rf {s['recall']} prec {s['precision']} survFC {s['surviving_false_cognates']}",
                  file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


# ---------------------------------------------------------------------------------------
# Phase 2: Act 1 alarm/anomaly co-reference (instance)
# ---------------------------------------------------------------------------------------
P2_COLS = ["case", "phase", "model", "placement", "trial", "precision", "recall",
           "experiment_only_recall", "residual", "interrogate_calls", "reasoning_tokens", "latency_s"]


def phase2(models, trials, budget, fresh, write, save_transcripts=False):
    ecase = load_instance_case(ROOT / "benchmark" / "cases" / "obs_instance")
    path = ROOT / "results" / f"obs_instance_phase2{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P2_COLS, fresh) if write else (None, None, []))
    tdir = ROOT / "results" / "obs_instance_transcripts"
    if save_transcripts:
        tdir.mkdir(parents=True, exist_ok=True)
    done = {(r["model"], r["placement"], r["trial"]) for r in rows}
    plan = [(m, pl) for m in models for pl in SPECTRUM]
    total = len(plan) * trials
    i = 0
    for (m, pl) in plan:
        for t in range(trials):
            i += 1
            key = (m, pl, str(t))
            if key in done:
                banner(2, i, total, f"{m}/{pl}/t{t} (skip)"); continue
            banner(2, i, total, f"{m}/{pl}/trial {t+1}")
            stack = InstanceAgentStack(ecase, model=m, reference_variant="none",
                                       evidence=set(EVIDENCE_ALL), inert_side="b", budget=budget)
            try:
                rec = stack.reconcile(placement=pl)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True); continue
            proposed, correct = set(rec.proposed), ecase.correct_pairs
            eo = ecase.experiment_only_pairs
            tp = len(proposed & correct); fp = len(proposed - correct); fn = len(correct - proposed)
            row = {"case": "obs_instance", "phase": "coref", "model": m, "placement": pl, "trial": t,
                   "precision": round(tp/(tp+fp), 3) if (tp+fp) else 0.0,
                   "recall": round(tp/(tp+fn), 3) if (tp+fn) else 0.0,
                   "experiment_only_recall": round(len(proposed & eo)/len(eo), 3) if eo else "",
                   "residual": len(rec.residual_a)+len(rec.residual_b),
                   "interrogate_calls": rec.oracle_calls.get("interrogate", 0),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", "")}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            if save_transcripts:
                (tdir / f"obs_instance_{m}_{pl}_t{t}.json").write_text(
                    json.dumps({"model": m, "placement": pl, "trial": t, "row": row,
                                "transcript": rec.transcript}, indent=2))
            print(f"      prec {row['precision']} rec {row['recall']} eo {row['experiment_only_recall']}",
                  file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


# ---------------------------------------------------------------------------------------
# Phase 3: Act 2 pragmatic verdict
# ---------------------------------------------------------------------------------------
P3_COLS = ["case", "phase", "model", "pragmatics", "context", "trial", "n_anomalies",
           "verdict_correct", "verdict_accuracy", "false_page", "missed_suppress",
           "reasoning_tokens", "latency_s"]


def phase3(a2, gold, models, trials, fresh, write):
    anomalies, contexts = a2["anomalies"], a2["contexts"]
    path = ROOT / "results" / f"obs_{CASE}_phase3{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P3_COLS, fresh) if write else (None, None, []))
    done = {(r["model"], r["pragmatics"], r["context"], r["trial"]) for r in rows}
    plan = [(m, prag, ctx) for m in models for prag in ("on", "off") for ctx in contexts]
    total = len(plan) * trials
    i = 0
    for (m, prag, ctx) in plan:
        for t in range(trials):
            i += 1
            key = (m, prag, ctx["id"], str(t))
            if key in done:
                banner(3, i, total, f"{m}/prag={prag}/{ctx['id']}/t{t} (skip)"); continue
            banner(3, i, total, f"{m}/prag={prag}/{ctx['id']}/trial {t+1}")
            stack = VerdictStack(model=m, pragmatics=(prag == "on"))
            try:
                rec = stack.reconcile(anomalies, ctx)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True); continue
            correct = fp = missed = 0
            for a in anomalies:
                want = gold["verdicts"][a["id"]][ctx["id"]]
                got = rec.verdicts.get(a["id"], "?")
                correct += int(got == want)
                if want in ("suppress", "watch") and got == "act":
                    fp += 1                      # a false page (acted where it should hold/suppress)
                if want == "suppress" and got != "suppress":
                    missed += 1
            n = len(anomalies)
            row = {"case": CASE, "phase": "verdict", "model": m, "pragmatics": prag,
                   "context": ctx["id"], "trial": t, "n_anomalies": n, "verdict_correct": correct,
                   "verdict_accuracy": round(correct/n, 3) if n else 0.0, "false_page": fp,
                   "missed_suppress": missed, "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", "")}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      acc {row['verdict_accuracy']} false-pages {fp} missed-suppress {missed}",
                  file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


# ---------------------------------------------------------------------------------------
# Phase 4: Act 2 multi-symptom correlation
# ---------------------------------------------------------------------------------------
P4_COLS = ["case", "phase", "model", "pragmatics", "scenario", "trial", "n_incidents_gold",
           "n_incidents_agent", "partition_exact", "cause_correct", "cause_accuracy",
           "reasoning_tokens", "latency_s"]


def phase4(a2, gold, models, trials, fresh, write):
    scenarios = a2["scenarios"]
    path = ROOT / "results" / f"obs_{CASE}_phase4{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P4_COLS, fresh) if write else (None, None, []))
    done = {(r["model"], r["pragmatics"], r["scenario"], r["trial"]) for r in rows}
    plan = [(m, prag, sc) for m in models for prag in ("on", "off") for sc in scenarios]
    total = len(plan) * trials
    i = 0
    for (m, prag, sc) in plan:
        gincs = gold["incidents"][sc["id"]]
        gparts = {frozenset(g["symptoms"]) for g in gincs}
        gcause = {frozenset(g["symptoms"]): g["cause"] for g in gincs}
        for t in range(trials):
            i += 1
            key = (m, prag, sc["id"], str(t))
            if key in done:
                banner(4, i, total, f"{m}/prag={prag}/{sc['id']}/t{t} (skip)"); continue
            banner(4, i, total, f"{m}/prag={prag}/{sc['id']}/trial {t+1}")
            stack = CorrelationStack(model=m, pragmatics=(prag == "on"))
            try:
                rec = stack.reconcile(sc)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True); continue
            aparts = {frozenset(a["symptoms"]) for a in rec.incidents}
            acause = {frozenset(a["symptoms"]): a["cause"] for a in rec.incidents}
            partition_exact = (aparts == gparts)
            cause_correct = sum(1 for p in gparts if p in acause and acause[p] == gcause[p])
            row = {"case": CASE, "phase": "correlation", "model": m, "pragmatics": prag,
                   "scenario": sc["id"], "trial": t, "n_incidents_gold": len(gincs),
                   "n_incidents_agent": len(rec.incidents), "partition_exact": partition_exact,
                   "cause_correct": cause_correct,
                   "cause_accuracy": round(cause_correct/len(gincs), 3) if gincs else 0.0,
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", "")}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      partition-exact {partition_exact} cause-acc {row['cause_accuracy']} "
                  f"(agent {len(rec.incidents)} vs gold {len(gincs)} incidents)", file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | 1 | 2 | 3 | 4")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--budget", type=int, default=20, help="instance-oracle call cap (phase 2)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (default: resume)")
    ap.add_argument("--save-transcripts", action="store_true",
                    help="phase 2: save a per-cell co-reference transcript to results/obs_instance_transcripts/")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr); return 2

    case = Case.load(ROOT / "benchmark" / "cases" / CASE)
    a2 = load_act2(ROOT / "benchmark" / "cases" / CASE)
    gold_path = ROOT / "benchmark" / "cases" / CASE / "obs_act2_gold.json"
    if not gold_path.exists():
        print("no obs_act2_gold.json; run benchmark/derive_observability_gold.py first", file=sys.stderr)
        return 2
    gold = json.loads(gold_path.read_text())
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = max(1, args.trials)
    write = not args.no_write
    if write:
        (ROOT / "results").mkdir(exist_ok=True)

    stages = ["1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "1":
            phase1(case, models, trials, args.fresh, write)
        elif st == "2":
            phase2(models, trials, args.budget, args.fresh, write, save_transcripts=args.save_transcripts)
        elif st == "3":
            phase3(a2, gold, models, trials, args.fresh, write)
        elif st == "4":
            phase4(a2, gold, models, trials, args.fresh, write)
        else:
            print(f"unknown stage '{st}'", file=sys.stderr); return 2
    print("observability study: done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
