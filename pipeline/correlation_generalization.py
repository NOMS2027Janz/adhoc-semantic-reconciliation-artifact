#!/usr/bin/env python3
"""E7b -- does composition & correlation generalise beyond observability? (Brad pt 6).

E7 established that the correlation operation is unconditionally needed WITHIN the observability
setting (turn it off and the weaker models cannot group symptoms into incidents at all). E7b tests
whether the SAME operation carries over to a setting where the study never performs it: the
CONFIGURATION / transport domain. The case `correlation_config` poses multi-symptom cross-layer
correlation over the configuration setting's own entities (an OTN line underlies an IP link
underlies an L3VPN/EVPN service), scored the same way as observability phase 4.

It reuses the exact machinery under test -- the `CorrelationStack` agent and the `correlate()`
oracle (via the derived gold) -- so a positive result means the operation is not an artefact of the
observability case. Pragmatics ON (semantics + dependency map) vs OFF (the legacy page-everything
baseline), across the capability ladder.

    python pipeline/correlation_generalization.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/correlation_generalization.py --smoke        # one model, one trial

Writes results/correlation_config.csv (one row per model x pragmatics x scenario x trial).
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
from reconcile.stacks.agent_observability import CorrelationStack   # noqa: E402
from run import load_dotenv                                         # noqa: E402

CASE = "correlation_config"
COLS = ["case", "model", "pragmatics", "scenario", "trial", "n_incidents_gold",
        "n_incidents_agent", "partition_exact", "cause_correct", "cause_accuracy",
        "reasoning_tokens", "latency_s"]


def load_case():
    cdir = ROOT / "benchmark" / "cases" / CASE
    scenarios = json.loads((cdir / "scenarios.json").read_text())["scenarios"]
    gold_path = cdir / "corr_gold.json"
    if not gold_path.exists():
        print("no corr_gold.json; run benchmark/derive_correlation_config_gold.py first", file=sys.stderr)
        sys.exit(2)
    gold = json.loads(gold_path.read_text())["incidents"]
    return scenarios, gold


def run_key(r):
    return (str(r["model"]), r["pragmatics"], r["scenario"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    scenarios, gold = load_case()
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / f"{CASE}.csv"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        rows = list(csv.DictReader(out.open(newline="")))
        done = {run_key(r) for r in rows}
        print(f"Resuming: {len(done)} rows in {out.name}", file=sys.stderr)

    fh = writer = None
    if write:
        fresh = args.fresh or not out.exists()
        fh = out.open("w" if fresh else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLS)
        if fresh:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    def emit(row):
        rows.append(row); done.add(run_key(row))
        if writer is not None:
            writer.writerow(row); fh.flush()

    plan = [(m, prag, sc) for m in models for prag in ("on", "off") for sc in scenarios]
    total = len(plan) * trials
    i = 0
    for (m, prag, sc) in plan:
        gincs = gold[sc["id"]]
        gparts = {frozenset(g["symptoms"]) for g in gincs}
        gcause = {frozenset(g["symptoms"]): g["cause"] for g in gincs}
        for t in range(trials):
            i += 1
            key = (m, prag, sc["id"], str(t))
            if key in done:
                print(f"[{i}/{total}] {m}/prag={prag}/{sc['id']}/t{t} (skip)", file=sys.stderr)
                continue
            print(f"[{i}/{total}] {m}/prag={prag}/{sc['id']}/trial {t+1}", file=sys.stderr, flush=True)
            stack = CorrelationStack(model=m, pragmatics=(prag == "on"))
            try:
                rec = stack.reconcile(sc)
            except Exception as e:  # noqa: BLE001
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            aparts = {frozenset(a["symptoms"]) for a in rec.incidents}
            acause = {frozenset(a["symptoms"]): a["cause"] for a in rec.incidents}
            partition_exact = (aparts == gparts)
            cause_correct = sum(1 for p in gparts if p in acause and acause[p] == gcause[p])
            row = {"case": CASE, "model": m, "pragmatics": prag, "scenario": sc["id"], "trial": t,
                   "n_incidents_gold": len(gincs), "n_incidents_agent": len(rec.incidents),
                   "partition_exact": partition_exact, "cause_correct": cause_correct,
                   "cause_accuracy": round(cause_correct / len(gincs), 3) if gincs else 0.0,
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", "")}
            emit(row)
            print(f"      partition-exact {partition_exact} cause-acc {row['cause_accuracy']} "
                  f"(agent {len(rec.incidents)} vs gold {len(gincs)} incidents)", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    if not rows:
        return
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])

    def frac_exact(rs):
        xs = [1.0 if str(r["partition_exact"]).lower() == "true" else 0.0 for r in rs]
        return sum(xs) / len(xs) if xs else 0.0

    def mean_cause(rs):
        xs = [float(r["cause_accuracy"]) for r in rs if r["cause_accuracy"] not in ("", None)]
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== E7b  Correlation ported to the configuration domain: ON vs OFF, across the ladder ===")
    print(f"    {'model':13}{'pragmatics':12}{'partition-exact':17}{'cause-accuracy':15}")
    for m in models:
        for prag in ("on", "off"):
            rs = [r for r in rows if r["model"] == m and r["pragmatics"] == prag]
            if not rs:
                continue
            print(f"    {m:13}{prag:12}{frac_exact(rs):<17.2f}{mean_cause(rs):<15.2f}")
    print("\n    Reading: if ON >> OFF here as it did in observability, the correlation operation")
    print("    generalises beyond the setting it was authored in -- it is a real operation, not an")
    print("    artefact of the observability case.")


if __name__ == "__main__":
    raise SystemExit(main())
