#!/usr/bin/env python3
"""Single-command staged runner for the intent setting (the second operational setting).

    python intent_study.py --stage all            # launch-and-leave: all four phases in sequence
    python intent_study.py --stage 2 --model gpt-5-mini   # one phase, one model (parallel terminals)

Four phases run in sequence, each writing its own segmented CSV and printing a phase banner
with live progress, checkpointed and resumable (re-run to continue where it left off):

  Phase 1/4  refine-down + satisfaction   — spectrum x reference arm x model
  Phase 2/4  feasibility & the judge      — policy x spectrum(+ mute vs pre-placed policy) x model
  Phase 3/4  endpoint co-reference        — reuses the instance agent on the intent endpoints
  Phase 4/4  assure-up (lifecycle)        — the multi-hop trajectories; full transcripts retained

Consistent with the illustration-first design, the measured cross-product is modest and
phase 4 runs a few crafted multi-hop scenarios end to end with their transcripts kept.
Correctness is the currency; oracle calls, turns, and tokens are effort.
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
from reconcile.intent import (load_intent_case, violated_bounds, satisfies_all)   # noqa: E402
from reconcile.stacks.agent_intent import IntentAgentStack                        # noqa: E402
from reconcile.instance import load_instance_case                                 # noqa: E402
from reconcile.stacks.agent_instance import InstanceAgentStack, EVIDENCE_ALL      # noqa: E402
from run import load_dotenv                                                       # noqa: E402

# placement specs: (label, placement, inert_side, consumer_mode)
SPECTRUM_SIMPLE = [
    ("both_cognitive", "both_cognitive", "n", "policy"),
    ("provider_inert", "one_inert", "n", "policy"),
    ("both_inert", "both_inert", "n", "policy"),
]
SPECTRUM_FULL = [
    ("both_cognitive", "both_cognitive", "n", "policy"),
    ("provider_inert", "one_inert", "n", "policy"),
    ("consumer_policy", "one_inert", "o", "policy"),      # consumer inert but pre-placed policy decides
    ("consumer_mute", "one_inert", "o", "mute"),          # consumer inert and mute: must refer
    ("both_inert", "both_inert", "n", "policy"),
]
REFERENCES = ["none", "unit", "invariant"]
INERT_SPECS = [SPECTRUM_FULL[1], SPECTRUM_FULL[4]]   # provider_inert, both_inert
PHASE_LABEL = {1: "refine-down + satisfaction", 2: "feasibility & the judge (negotiation)",
               3: "endpoint co-reference", 4: "assure-up (fulfilment lifecycle)",
               5: "negotiation reference sweep (inert placements)"}


def banner(pnum, i, total, detail):
    print(f"=== Phase {pnum}/4 · {PHASE_LABEL[pnum]} ===   [{i:>3}/{total}]  {detail}",
          file=sys.stderr, flush=True)


def open_csv(path, columns, fresh):
    rows, done = [], set()
    if path.exists() and not fresh:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    fresh_file = fresh or not path.exists()
    fh = path.open("w" if fresh_file else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=columns)
    if fresh_file:
        w.writeheader(); fh.flush(); rows = []
    return fh, w, rows


# ---------------------------------------------------------------------------------------
# Phase 1: refine-down + satisfaction
# ---------------------------------------------------------------------------------------
P1_COLS = ["case", "phase", "model", "placement", "reference", "trial", "submitted",
           "n_intents", "satisfaction_correct", "satisfaction_accuracy", "eo_correct",
           "eo_total", "fabricated", "feasibility_calls", "turns", "total_tokens",
           "reasoning_tokens", "latency_s"]


def score_refine(case, rec):
    gold = case.gold["refine"]
    rbyid = case.realisation_by_id
    correct = eo_correct = eo_total = fabricated = 0
    for r in rec.refinements:
        iid = r.get("intent_id")
        g = gold.get(iid)
        if not g:
            continue
        sat, rid = bool(r.get("satisfies")), r.get("realisation_id", "")
        want = g["satisfiable"]
        ok = (sat == want)
        if want and rid not in g["actual_satisfiers"]:
            ok = False
        if not want and rid:
            ok = False
        if rid and (rid not in rbyid or rbyid[rid].get("pair") != g["pair"]):
            fabricated += 1
            ok = False
        correct += int(ok)
        if g["experiment_only"]:
            eo_total += 1
            eo_correct += int(ok)
    n = len(gold)
    return {"n_intents": n, "satisfaction_correct": correct,
            "satisfaction_accuracy": round(correct / n, 3) if n else 0.0,
            "eo_correct": eo_correct, "eo_total": eo_total, "fabricated": fabricated}


def phase1(case, models, trials, budget, fresh, write, pnum=1):
    path = ROOT / "results" / f"intent_{case.name}_phase1{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P1_COLS, fresh) if write else (None, None, []))
    done = {p1key(r) for r in rows}
    plan = [(m, spec, ref) for m in models for spec in SPECTRUM_SIMPLE for ref in REFERENCES]
    total = len(plan) * trials
    i = 0
    for (m, spec, ref) in plan:
        label, placement, inert_side, cmode = spec
        for t in range(trials):
            i += 1
            key = (m, label, ref, str(t))
            if key in done:
                banner(pnum, i, total, f"{m}/{label}/{ref}/t{t}  (skip)")
                continue
            banner(pnum, i, total, f"{m}/{label}/ref={ref}/trial {t+1}")
            stack = IntentAgentStack(case, model=m, phase="refine", reference_variant=ref,
                                     inert_side=inert_side, consumer_mode=cmode, budget=budget)
            try:
                rec = stack.reconcile(placement=placement)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score_refine(case, rec)
            row = {"case": case.name, "phase": "refine", "model": m, "placement": label,
                   "reference": ref, "trial": t, "submitted": rec.submitted,
                   "feasibility_calls": rec.oracle_calls.get("feasibility", 0),
                   "turns": rec.effort.get("turns", ""), "total_tokens": rec.effort.get("total_tokens", ""),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      sat-acc {s['satisfaction_accuracy']} eo {s['eo_correct']}/{s['eo_total']} "
                  f"fab {s['fabricated']} feas {row['feasibility_calls']}", file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


def p1key(r):
    return (str(r["model"]), r["placement"], r["reference"], str(r["trial"]))


# ---------------------------------------------------------------------------------------
# Phase 2: feasibility & the judge (negotiation)
# ---------------------------------------------------------------------------------------
P2_COLS = ["case", "phase", "model", "policy", "placement", "reference", "trial", "submitted",
           "n_intents", "decision_correct", "decision_accuracy", "offer_correct", "refer_count",
           "best_calls", "policy_calls", "feasibility_calls", "turns", "total_tokens",
           "reasoning_tokens", "latency_s"]


def score_negotiate(case, rec, policy_id):
    gold = case.gold["negotiation"]
    neg = case.gold["negotiation_intents"]
    dec_correct = offer_correct = refer = 0
    for d in rec.decisions:
        iid = d.get("intent_id")
        if iid not in gold:
            continue
        g = gold[iid][policy_id]
        agent_dec = d.get("decision")
        if agent_dec == "refer":
            refer += 1
        if agent_dec == g["decision"]:
            dec_correct += 1
            if d.get("offer_realisation") == g["best_offer"]:
                offer_correct += 1
    n = len(neg)
    return {"n_intents": n, "decision_correct": dec_correct,
            "decision_accuracy": round(dec_correct / n, 3) if n else 0.0,
            "offer_correct": offer_correct, "refer_count": refer}


def phase2(case, models, trials, budget, fresh, write, pnum=2):
    path = ROOT / "results" / f"intent_{case.name}_phase2{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P2_COLS, fresh) if write else (None, None, []))
    done = {p2key(r) for r in rows}
    policies = [p["id"] for p in case.policies]
    # main grid: full spectrum x policy at reference=none; plus reference effect at both_cognitive
    plan = []
    for m in models:
        for pol in policies:
            for spec in SPECTRUM_FULL:
                plan.append((m, pol, spec, "none"))
            for ref in ("unit", "invariant"):
                plan.append((m, pol, SPECTRUM_FULL[0], ref))   # reference effect, both_cognitive
    total = len(plan) * trials
    i = 0
    for (m, pol, spec, ref) in plan:
        label, placement, inert_side, cmode = spec
        for t in range(trials):
            i += 1
            key = (m, pol, label, ref, str(t))
            if key in done:
                banner(pnum, i, total, f"{m}/{pol}/{label}/{ref}/t{t}  (skip)")
                continue
            banner(pnum, i, total, f"{m}/{pol}/{label}/ref={ref}/trial {t+1}")
            stack = IntentAgentStack(case, model=m, phase="negotiate", policy_id=pol,
                                     reference_variant=ref, inert_side=inert_side,
                                     consumer_mode=cmode, budget=budget)
            try:
                rec = stack.reconcile(placement=placement)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score_negotiate(case, rec, pol)
            row = {"case": case.name, "phase": "negotiate", "model": m, "policy": pol,
                   "placement": label, "reference": ref, "trial": t, "submitted": rec.submitted,
                   "best_calls": rec.oracle_calls.get("best_achievable", 0),
                   "policy_calls": rec.oracle_calls.get("policy", 0),
                   "feasibility_calls": rec.oracle_calls.get("feasibility", 0),
                   "turns": rec.effort.get("turns", ""), "total_tokens": rec.effort.get("total_tokens", ""),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      dec-acc {s['decision_accuracy']} offer {s['offer_correct']} "
                  f"refer {s['refer_count']} calls b{row['best_calls']}/p{row['policy_calls']}",
                  file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


def p2key(r):
    return (str(r["model"]), r["policy"], r["placement"], r["reference"], str(r["trial"]))


def phase2_refs(case, models, trials, budget, fresh, write, pnum=5):
    """Supplementary: reference arms (unit/invariant) at the INERT placements, where a
    published anchor could let an inert-side agent close what it otherwise refers. The
    baseline (reference=none) at these placements already exists in the phase-2 CSV."""
    path = ROOT / "results" / f"intent_{case.name}_phase2refs{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P2_COLS, fresh) if write else (None, None, []))
    done = {p2key(r) for r in rows}
    policies = [p["id"] for p in case.policies]
    plan = [(m, pol, spec, ref) for m in models for pol in policies
            for spec in INERT_SPECS for ref in ("unit", "invariant")]
    total = len(plan) * trials
    i = 0
    for (m, pol, spec, ref) in plan:
        label, placement, inert_side, cmode = spec
        for t in range(trials):
            i += 1
            key = (m, pol, label, ref, str(t))
            if key in done:
                banner(pnum, i, total, f"{m}/{pol}/{label}/{ref}/t{t}  (skip)")
                continue
            banner(pnum, i, total, f"{m}/{pol}/{label}/ref={ref}/trial {t+1}")
            stack = IntentAgentStack(case, model=m, phase="negotiate", policy_id=pol,
                                     reference_variant=ref, inert_side=inert_side,
                                     consumer_mode=cmode, budget=budget)
            try:
                rec = stack.reconcile(placement=placement)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score_negotiate(case, rec, pol)
            row = {"case": case.name, "phase": "negotiate", "model": m, "policy": pol,
                   "placement": label, "reference": ref, "trial": t, "submitted": rec.submitted,
                   "best_calls": rec.oracle_calls.get("best_achievable", 0),
                   "policy_calls": rec.oracle_calls.get("policy", 0),
                   "feasibility_calls": rec.oracle_calls.get("feasibility", 0),
                   "turns": rec.effort.get("turns", ""), "total_tokens": rec.effort.get("total_tokens", ""),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      dec-acc {s['decision_accuracy']} refer {s['refer_count']} "
                  f"calls b{row['best_calls']}/p{row['policy_calls']}", file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


# ---------------------------------------------------------------------------------------
# Phase 3: endpoint co-reference (reuse the instance agent)
# ---------------------------------------------------------------------------------------
P3_COLS = ["case", "phase", "model", "placement", "trial", "submitted", "precision", "recall",
           "experiment_only_recall", "residual_total", "interrogate_calls", "turns",
           "total_tokens", "reasoning_tokens", "latency_s"]


def score_coref(ecase, rec):
    proposed = set(rec.proposed)
    correct = ecase.correct_pairs
    eo = ecase.experiment_only_pairs
    tp = len(proposed & correct)
    fp = len(proposed - correct)
    fn = len(correct - proposed)
    eo_hit = proposed & eo
    return {"precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
            "experiment_only_recall": round(len(eo_hit) / len(eo), 3) if eo else "",
            "residual_total": len(rec.residual_a) + len(rec.residual_b)}


def phase3(models, trials, budget, fresh, write, pnum=3):
    ecase = load_instance_case(ROOT / "benchmark" / "cases" / "intent_endpoints")
    path = ROOT / "results" / f"intent_endpoints_phase3{suffix(models)}.csv"
    fh, w, rows = (open_csv(path, P3_COLS, fresh) if write else (None, None, []))
    done = {(str(r["model"]), r["placement"], str(r["trial"])) for r in rows}
    plan = [(m, spec) for m in models for spec in SPECTRUM_SIMPLE]
    total = len(plan) * trials
    i = 0
    for (m, spec) in plan:
        label, placement, inert_side, _ = spec
        iside = "b" if inert_side == "n" else "b"   # instance uses 'b' as inert side
        for t in range(trials):
            i += 1
            key = (m, label, str(t))
            if key in done:
                banner(pnum, i, total, f"{m}/{label}/t{t}  (skip)")
                continue
            banner(pnum, i, total, f"{m}/{label}/trial {t+1}")
            stack = InstanceAgentStack(ecase, model=m, reference_variant="none",
                                       evidence=set(EVIDENCE_ALL), inert_side="b", budget=budget)
            try:
                rec = stack.reconcile(placement=placement)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score_coref(ecase, rec)
            row = {"case": "intent_endpoints", "phase": "coref", "model": m, "placement": label,
                   "trial": t, "submitted": rec.submitted,
                   "interrogate_calls": rec.oracle_calls.get("interrogate", 0),
                   "turns": rec.effort.get("turns", ""), "total_tokens": rec.effort.get("total_tokens", ""),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            print(f"      prec {s['precision']} rec {s['recall']} eo {s['experiment_only_recall']} "
                  f"resid {s['residual_total']}", file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


# ---------------------------------------------------------------------------------------
# Phase 4: assure-up lifecycle (multi-hop trajectories, transcripts retained)
# ---------------------------------------------------------------------------------------
P4_COLS = ["case", "phase", "model", "trajectory", "policy", "placement", "trial", "setpiece",
           "submitted", "n_hops", "fulfilment_correct", "decision_correct", "migration_correct",
           "hop_accuracy", "telemetry_calls", "best_calls", "policy_calls", "turns",
           "total_tokens", "reasoning_tokens", "latency_s"]


def score_assure(case, rec, traj_id):
    tg = case.gold["lifecycle"][traj_id]["hops"]
    gmap = {h["hop_id"]: h for h in tg}
    ful = dec = mig = both = 0
    for h in rec.hops:
        g = gmap.get(h.get("hop_id"))
        if not g:
            continue
        f_ok = h.get("fulfilment") == g["fulfilment"]
        d_ok = h.get("decision") == g["decision"]
        m_ok = h.get("new_realisation") == g["new_realisation"]
        ful += int(f_ok); dec += int(d_ok); mig += int(m_ok)
        both += int(f_ok and d_ok and m_ok)
    n = len(tg)
    return {"n_hops": n, "fulfilment_correct": ful, "decision_correct": dec,
            "migration_correct": mig, "hop_accuracy": round(both / n, 3) if n else 0.0}


def phase4(case, models, trials, budget, fresh, write, pnum=4):
    path = ROOT / "results" / f"intent_{case.name}_phase4{suffix(models)}.csv"
    tdir = ROOT / "results" / "intent_transcripts"
    if write:
        tdir.mkdir(parents=True, exist_ok=True)
    fh, w, rows = (open_csv(path, P4_COLS, fresh) if write else (None, None, []))
    done = {(str(r["model"]), r["trajectory"], str(r["trial"])) for r in rows}
    trajs = case.lifecycle
    plan = [(m, tr) for m in models for tr in trajs]
    total = len(plan) * trials
    i = 0
    for (m, tr) in plan:
        for t in range(trials):
            i += 1
            key = (m, tr["id"], str(t))
            if key in done:
                banner(pnum, i, total, f"{m}/{tr['id']}/t{t}  (skip)")
                continue
            banner(pnum, i, total, f"{m}/{tr['id']} [{tr['policy_id']}]/trial {t+1}")
            stack = IntentAgentStack(case, model=m, phase="assure", policy_id=tr["policy_id"],
                                     trajectory_id=tr["id"], budget=budget)
            try:
                rec = stack.reconcile(placement="both_cognitive")
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score_assure(case, rec, tr["id"])
            row = {"case": case.name, "phase": "assure", "model": m, "trajectory": tr["id"],
                   "policy": tr["policy_id"], "placement": "both_cognitive", "trial": t,
                   "setpiece": bool(tr.get("setpiece")), "submitted": rec.submitted,
                   "telemetry_calls": rec.oracle_calls.get("telemetry", 0),
                   "best_calls": rec.oracle_calls.get("best_achievable", 0),
                   "policy_calls": rec.oracle_calls.get("policy", 0),
                   "turns": rec.effort.get("turns", ""), "total_tokens": rec.effort.get("total_tokens", ""),
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if w:
                w.writerow(row); fh.flush()
            # retain transcripts: always for the set-piece (trial 0), else keep the first per traj
            if write and t == 0:
                (tdir / f"{tr['id']}_{m}.json").write_text(json.dumps(
                    {"model": m, "trajectory": tr["id"], "policy": tr["policy_id"],
                     "setpiece": bool(tr.get("setpiece")), "row": row,
                     "transcript": rec.transcript}, indent=2))
            print(f"      hop-acc {s['hop_accuracy']} ful {s['fulfilment_correct']}/{s['n_hops']} "
                  f"dec {s['decision_correct']}/{s['n_hops']} mig {s['migration_correct']}/{s['n_hops']}",
                  file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return rows


def suffix(models):
    return f"_{models[0]}" if len(models) == 1 else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | 1 | 2 | 3 | 4 | 2refs")
    ap.add_argument("--case", default="intent_hard")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--budget", type=int, default=20, help="oracle call cap per run")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (default: resume)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case = load_intent_case(ROOT / "benchmark" / "cases" / args.case)
    if not case.gold:
        print("no intent_gold.json; run benchmark/derive_intent_gold.py first", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = max(1, args.trials)
    write = not args.no_write
    if write:
        (ROOT / "results").mkdir(exist_ok=True)

    stages = ["1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "1":
            phase1(case, models, trials, args.budget, args.fresh, write)
        elif st == "2":
            phase2(case, models, trials, args.budget, args.fresh, write)
        elif st == "3":
            phase3(models, trials, args.budget, args.fresh, write)
        elif st == "4":
            phase4(case, models, trials, args.budget, args.fresh, write)
        elif st == "2refs":
            phase2_refs(case, models, trials, args.budget, args.fresh, write)
        else:
            print(f"unknown stage '{st}'", file=sys.stderr)
            return 2
    print("intent study: done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
