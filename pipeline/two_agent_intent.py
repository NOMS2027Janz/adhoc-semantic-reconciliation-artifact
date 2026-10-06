#!/usr/bin/env python3
"""E13 -- a real second reasoning agent in the intent negotiation (Brad pt 8, optional).

The intent 'negotiate' phase is CONSUMER-agent vs provider-ORACLES: for each infeasible intent the
deterministic `best_achievable` oracle computes the provider's offer, and the consumer decides
accept/reject by policy. E13 replaces that oracle with an actual reasoning PROVIDER AGENT: given the
intent's bounds, its candidate realisations (with their real delivered attributes) and the policy's
priority order, the provider agent chooses which realisation to OFFER. The consumer decision is held
deterministic (the policy), so any change in outcome is attributable to making the provider a second
agent -- the test of whether the two-agent mechanism changes anything.

Two arms per (intent, policy):
  * oracle -- offer = best_achievable(...) (deterministic; equals the gold offer). A control.
  * agent  -- offer = the provider agent's chosen realisation, across the capability ladder.

Both arms take the same consumer step (policy_decision on the offer). Scored vs the negotiation gold:
does the provider agent reproduce the oracle's best-achievable offer, and does the final decision
still match gold?

    python pipeline/two_agent_intent.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/two_agent_intent.py --smoke

Writes results/two_agent_intent.csv (one row per arm x model x intent x policy x trial).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.intent import (load_intent_case, best_achievable,   # noqa: E402
                              policy_decision, violated_bounds)
from run import load_dotenv                                        # noqa: E402

CASE = "intent_hard"
COLS = ["case", "arm", "model", "intent", "policy", "trial",
        "offer_id", "gold_offer", "offer_correct", "decision", "gold_decision", "decision_correct"]

PROVIDER_SYSTEM = (
    "You are the PROVIDER in an intent negotiation. A consumer intent states required BOUNDS "
    "(bandwidth_min in Gbps, latency_max in ms, availability_min as a fraction, protection_required). "
    "None of your candidate realisations meets every bound. Each realisation lists its REAL delivered "
    "attributes (bw_gbps, latency_ms, availability, protection, cost). You are given a PRIORITY order "
    "over the bound kinds, highest priority first. Offer the ONE realisation that is best-achievable: "
    "it must preserve the highest-priority bounds and degrade only the LOWEST-priority bound(s). Among "
    "ties, prefer the lower cost. Return the id of the realisation you offer."
)


class Offer(BaseModel):
    realisation_id: str


def provider_agent_offer(bounds, options, priority, model, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    user = {
        "intent_bounds": bounds,
        "priority_high_to_low": priority,
        "candidate_realisations": [{"id": o["id"], "attrs": o["attrs"]} for o in options],
    }
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": PROVIDER_SYSTEM},
                  {"role": "user", "content": json.dumps(user, indent=2)}],
        response_format=Offer,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.realisation_id if parsed else None


def run_key(r):
    return (r["arm"], str(r["model"]), r["intent"], r["policy"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    case = load_intent_case(ROOT / "benchmark" / "cases" / CASE)
    intents = {i["id"]: i for i in json.loads((ROOT / "benchmark" / "cases" / CASE / "intents.json").read_text())["intents"]}
    policies = {p["id"]: p for p in json.loads((ROOT / "benchmark" / "cases" / CASE / "policies.json").read_text())["policies"]}
    gold = case.gold["negotiation"]
    neg_ids = case.gold["negotiation_intents"]

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    # options (actual attrs) per intent, as the gold uses
    options_by_intent = {iid: case.options_for(intents[iid]["pair"], actual=True) for iid in neg_ids}
    opt_by_id = {iid: {o["id"]: o for o in options_by_intent[iid]} for iid in neg_ids}

    need_api = True
    load_dotenv()
    if need_api and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / "two_agent_intent.csv"
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
        full = {c: row.get(c, "") for c in COLS}
        rows.append(full); done.add(run_key(full))
        if writer is not None:
            writer.writerow(full); fh.flush()

    def record(arm, model, iid, pid, trial, offer_id):
        it, pol = intents[iid], policies[pid]
        opt = opt_by_id[iid].get(offer_id)
        if opt is None:
            decision = "reject"
        else:
            decision = policy_decision(it["bounds"], opt["attrs"], pol)
        g = gold[iid][pid]
        emit({"case": CASE, "arm": arm, "model": model, "intent": iid, "policy": pid, "trial": trial,
              "offer_id": offer_id, "gold_offer": g["best_offer"],
              "offer_correct": int(offer_id == g["best_offer"]),
              "decision": decision, "gold_decision": g["decision"],
              "decision_correct": int(decision == g["decision"])})

    # oracle control arm (deterministic; model-independent, trial 0)
    for iid in neg_ids:
        for pid, pol in policies.items():
            key = ("oracle", "deterministic", iid, pid, "0")
            if key in done:
                continue
            best = best_achievable(intents[iid]["bounds"], options_by_intent[iid], pol["priority"])
            record("oracle", "deterministic", iid, pid, 0, best["id"] if best else None)

    # provider-agent arm
    total = len(models) * len(neg_ids) * len(policies) * trials
    i = 0
    for model in models:
        for iid in neg_ids:
            for pid, pol in policies.items():
                for t in range(trials):
                    i += 1
                    key = ("agent", str(model), iid, pid, str(t))
                    if key in done:
                        continue
                    print(f"[{i}/{total}] agent {model}/{iid}/{pid}/t{t+1}", file=sys.stderr, flush=True)
                    try:
                        offer = provider_agent_offer(intents[iid]["bounds"], options_by_intent[iid],
                                                     pol["priority"], model)
                    except Exception as e:  # noqa: BLE001
                        print(f"      ! {e}", file=sys.stderr, flush=True)
                        continue
                    record("agent", model, iid, pid, t, offer)

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    ag = [r for r in rows if r["arm"] == "agent"]
    if not ag:
        return
    models = []
    for r in ag:
        if r["model"] not in models:
            models.append(r["model"])

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, k):
        xs = [num(r[k]) for r in rs if num(r.get(k)) is not None]
        return sum(xs) / len(xs) if xs else None

    orc = [r for r in rows if r["arm"] == "oracle"]
    print("\n=== E13  Provider as a reasoning agent vs the best-achievable oracle ===")
    if orc:
        print(f"    oracle control: offer_correct {mean(orc,'offer_correct'):.2f} "
              f"decision_correct {mean(orc,'decision_correct'):.2f} (should be 1.00 / 1.00)")
    print(f"    {'model':13}{'offer matches oracle':22}{'decision matches gold':22}")
    for m in models:
        rs = [r for r in ag if r["model"] == m]
        print(f"    {m:13}{mean(rs,'offer_correct'):<22.2f}{mean(rs,'decision_correct'):<22.2f}")
    print("\n    Reading: if the provider agent reproduces the oracle's offers (offer≈1.0), the")
    print("    two-agent mechanism changes nothing and the oracle was a faithful stand-in; where it")
    print("    diverges (esp. weaker models), the provider agent negotiates differently -- and whether")
    print("    the DECISION still matches gold shows if that divergence actually changes outcomes.")


if __name__ == "__main__":
    raise SystemExit(main())
