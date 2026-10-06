#!/usr/bin/env python3
"""E15 -- a deliverability-reasoning provider: where the second agent starts to matter.

E13 put a real reasoning agent in the provider's seat and found it changed almost nothing: the
provider's task there is a solved optimisation over a fixed catalogue (offer the best-achievable
realisation under a priority order), so a reasoning agent just reproduces the deterministic oracle.
The honest scope of that null is that it holds *where the provider's side reduces to selecting from a
fixed menu*. A real provider also has to judge whether it can actually DELIVER a realisation against
its own network state, and may decline or counter-propose specific changes -- which is live cognition
on the provider side.

E15 adds exactly that. Each realisation runs on a bearer PATH; a capacity SCENARIO can saturate paths,
so an attribute-attractive realisation may be undeliverable right now. The provider must offer the
best-achievable realisation ON AN AVAILABLE PATH, or decline when nothing is deliverable. Three arms:

  * catalogue_oracle  -- best_achievable IGNORING capacity (the E13 oracle: a naive provider that may
                         offer an undeliverable service).
  * deliverability_oracle -- best_achievable among DELIVERABLE realisations, else decline
                         (deterministic; the gold).
  * agent             -- a reasoning provider given the bounds, the realisations WITH their path, and
                         the live capacity state, across the capability ladder.

Scenarios per intent: `clear` (no path saturated -> deliverability-best == catalogue-best, the E13
regime, so a correct agent should NOT diverge), `constrained` (the attractive realisation's path is
saturated -> a correct agent must counter-propose a deliverable alternative or accept the changed
outcome), and `outage` (every path saturated -> the provider must decline).

The question E15 answers: does a reasoning provider DIVERGE from the catalogue oracle once deliverability
binds (i.e. does the second agent start to matter), and is that divergence capability-gated?

    python pipeline/deliverability_negotiation.py --models gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/deliverability_negotiation.py --validate    # offline: gold + scenarios, no API

Writes results/deliverability_negotiation.csv (one row per arm x model x intent x scenario x policy x trial).
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
from reconcile.intent import (load_intent_case, best_achievable,   # noqa: E402
                              policy_decision, violated_bounds)
from run import load_dotenv                                        # noqa: E402

CASE = "intent_hard"

# The three negotiation intents (no realisation fully satisfies), and, per intent, the capacity
# scenarios: which realisation-bearing paths are saturated (undeliverable) in that state. A path is
# named for the realisation it carries; saturating it removes that realisation from what can be
# delivered now. Scenarios are policy-independent (a property of the network, not the consumer).
SCENARIOS = {
    "I2": {"clear": [], "constrained": ["r1b"], "outage": ["r1a", "r1b", "r1c"]},
    "I3": {"clear": [], "constrained": ["r4b"], "outage": ["r4a", "r4b"]},
    "I7": {"clear": [], "constrained": ["r2b"], "outage": ["r2a", "r2b"]},
}
NEG_INTENTS = list(SCENARIOS.keys())

COLS = ["case", "arm", "model", "intent", "scenario", "policy", "trial",
        "offer_id", "gold_offer", "offer_correct", "decision", "gold_decision", "decision_correct",
        "offer_deliverable", "catalogue_offer", "catalogue_undeliverable", "diverges_from_catalogue"]

DECLINE = "decline"

PROVIDER_SYSTEM = (
    "You are the PROVIDER in an intent negotiation, and you must decide what you can actually DELIVER "
    "right now. A consumer intent states required BOUNDS (bandwidth_min in Gbps, latency_max in ms, "
    "availability_min as a fraction, protection_required). None of your candidate realisations meets "
    "every bound. Each realisation lists its REAL delivered attributes (bw_gbps, latency_ms, "
    "availability, protection, cost) and the bearer PATH it runs on. You also hold a live NETWORK STATE "
    "saying which paths are AVAILABLE and which are SATURATED (at capacity). A realisation on a "
    "saturated path CANNOT be delivered now. You are given a PRIORITY order over the bound kinds, "
    "highest first. Offer the ONE realisation that is best-achievable AMONG THE DELIVERABLE ones (on an "
    "available path): it should preserve the highest-priority bounds and degrade only the lowest, and "
    "among ties prefer lower cost. If NO realisation is deliverable (every path saturated), you must "
    f"decline. Return the id of the realisation you offer, or \"{DECLINE}\" if you must decline."
)


def deliverable_options(options, saturated):
    return [o for o in options if o["id"] not in saturated]


def deliverability_gold(bounds, options, priority, saturated, policy):
    """Deterministic deliverability-aware provider: best_achievable among deliverable options; decline
    if none deliverable. Returns (offer_id_or_DECLINE, decision)."""
    deliver = deliverable_options(options, saturated)
    if not deliver:
        return DECLINE, DECLINE
    best = best_achievable(bounds, deliver, priority)
    return best["id"], policy_decision(bounds, best["attrs"], policy)


def catalogue_offer(bounds, options, priority):
    """The E13 oracle: best_achievable ignoring capacity."""
    best = best_achievable(bounds, options, priority)
    return best["id"] if best else DECLINE


def provider_agent_offer(bounds, options, priority, saturated, model, client=None):
    from pydantic import BaseModel

    class Offer(BaseModel):
        realisation_id: str

    if client is None:
        from openai import OpenAI
        client = OpenAI()
    net_state = [{"id": o["id"], "path": f"path::{o['id']}",
                  "available": o["id"] not in saturated} for o in options]
    user = {
        "intent_bounds": bounds,
        "priority_high_to_low": priority,
        "candidate_realisations": [{"id": o["id"], "path": f"path::{o['id']}", "attrs": o["attrs"]}
                                   for o in options],
        "network_state": net_state,
    }
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": PROVIDER_SYSTEM},
                  {"role": "user", "content": json.dumps(user, indent=2)}],
        response_format=Offer,
    )
    parsed = completion.choices[0].message.parsed
    return (parsed.realisation_id or DECLINE).strip() if parsed else DECLINE


def run_key(r):
    return (r["arm"], str(r["model"]), r["intent"], r["scenario"], r["policy"], str(r["trial"]))


def _decision_for(offer_id, bounds, opt_by_id, policy, saturated):
    """The consumer decision given a provider offer id (or DECLINE). An offer on a saturated path is
    undeliverable: the honest outcome is decline, not a phantom accept."""
    if offer_id == DECLINE or offer_id not in opt_by_id:
        return DECLINE
    if offer_id in saturated:
        return DECLINE  # provider named an undeliverable realisation; it cannot be delivered
    return policy_decision(bounds, opt_by_id[offer_id]["attrs"], policy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="results/deliverability_negotiation.csv",
                    help="output CSV (give each parallel terminal its own file, then merge)")
    ap.add_argument("--validate", action="store_true", help="offline: gold + scenario check, no API")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    case = load_intent_case(ROOT / "benchmark" / "cases" / CASE)
    intents = case.intent_by_id
    policies = case.policy_by_id
    options_by_intent = {iid: case.options_for(intents[iid]["pair"], actual=True) for iid in NEG_INTENTS}
    opt_by_id = {iid: {o["id"]: o for o in options_by_intent[iid]} for iid in NEG_INTENTS}

    if args.validate:
        return validate(intents, policies, options_by_intent)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    trials = max(1, args.trials)

    out = ROOT / args.out
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

    def record(arm, model, iid, scen, pid, trial, offer_id):
        it, pol = intents[iid], policies[pid]
        bounds, priority = it["bounds"], pol["priority"]
        opts = options_by_intent[iid]
        saturated = set(SCENARIOS[iid][scen])
        gold_offer, gold_decision = deliverability_gold(bounds, opts, priority, saturated, pol)
        cat_offer = catalogue_offer(bounds, opts, priority)
        cat_undeliverable = int(cat_offer in saturated)
        decision = _decision_for(offer_id, bounds, opt_by_id[iid], pol, saturated)
        emit({"case": CASE, "arm": arm, "model": model, "intent": iid, "scenario": scen,
              "policy": pid, "trial": trial,
              "offer_id": offer_id, "gold_offer": gold_offer,
              "offer_correct": int(offer_id == gold_offer),
              "decision": decision, "gold_decision": gold_decision,
              "decision_correct": int(decision == gold_decision),
              "offer_deliverable": int(offer_id == DECLINE or (offer_id in opt_by_id[iid]
                                       and offer_id not in saturated)),
              "catalogue_offer": cat_offer, "catalogue_undeliverable": cat_undeliverable,
              "diverges_from_catalogue": int(offer_id != cat_offer)})

    # deterministic control arms (model-independent, trial 0): catalogue oracle and deliverability oracle
    for iid in NEG_INTENTS:
        for scen in SCENARIOS[iid]:
            for pid, pol in policies.items():
                bounds, priority = intents[iid]["bounds"], pol["priority"]
                opts = options_by_intent[iid]
                saturated = set(SCENARIOS[iid][scen])
                if ("catalogue_oracle", "deterministic", iid, scen, pid, "0") not in done:
                    record("catalogue_oracle", "deterministic", iid, scen, pid, 0,
                           catalogue_offer(bounds, opts, priority))
                if ("deliverability_oracle", "deterministic", iid, scen, pid, "0") not in done:
                    g_off, _ = deliverability_gold(bounds, opts, priority, saturated, pol)
                    record("deliverability_oracle", "deterministic", iid, scen, pid, 0, g_off)

    total = len(models) * sum(len(SCENARIOS[i]) for i in NEG_INTENTS) * len(policies) * trials
    i = 0
    for model in models:
        for iid in NEG_INTENTS:
            for scen in SCENARIOS[iid]:
                saturated = set(SCENARIOS[iid][scen])
                for pid, pol in policies.items():
                    for t in range(trials):
                        i += 1
                        key = ("agent", str(model), iid, scen, pid, str(t))
                        if key in done:
                            continue
                        print(f"[{i}/{total}] agent {model}/{iid}/{scen}/{pid}/t{t+1}",
                              file=sys.stderr, flush=True)
                        try:
                            offer = provider_agent_offer(intents[iid]["bounds"], options_by_intent[iid],
                                                         pol["priority"], saturated, model)
                        except Exception as e:  # noqa: BLE001
                            print(f"      ! {e}", file=sys.stderr, flush=True)
                            continue
                        record("agent", model, iid, scen, pid, t, offer)

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

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, k):
        xs = [num(r[k]) for r in rs if num(r.get(k)) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    models = []
    for r in ag:
        if r["model"] not in models:
            models.append(r["model"])

    # the catalogue oracle's undeliverable-offer rate (deterministic; the cost of not reasoning)
    cat = [r for r in rows if r["arm"] == "catalogue_oracle"]
    print("\n=== E15  A deliverability-reasoning provider vs the catalogue oracle ===")
    if cat:
        print(f"    catalogue oracle offers an UNDELIVERABLE realisation in "
              f"{mean(cat,'catalogue_undeliverable'):.2f} of cells "
              f"(all in the constrained/outage scenarios).")
    print(f"\n    {'model':13}{'offer=gold':12}{'decision=gold':15}{'diverges(all)':15}{'diverges(constr+outage)':24}")
    for m in models:
        rs = [r for r in ag if r["model"] == m]
        constr = [r for r in rs if r["scenario"] in ("constrained", "outage")]
        clear = [r for r in rs if r["scenario"] == "clear"]
        print(f"    {m:13}{mean(rs,'offer_correct'):<12.2f}{mean(rs,'decision_correct'):<15.2f}"
              f"{mean(rs,'diverges_from_catalogue'):<15.2f}{mean(constr,'diverges_from_catalogue'):<24.2f}")
        print(f"      (clear-scenario divergence {mean(clear,'diverges_from_catalogue'):.2f} "
              f"-- should be ~0, the E13 regime; deliverable offers {mean(rs,'offer_deliverable'):.2f})")
    print("\n    Reading: in the clear scenario the provider's task is the E13 optimisation and a correct")
    print("    agent should track the catalogue oracle (divergence ~0). Once a path saturates, the")
    print("    deliverability-aware provider must diverge -- counter-propose or decline -- and whether it")
    print("    does, and matches the deliverability gold, is the capability-gated measure of the second")
    print("    agent starting to matter.")


def validate(intents, policies, options_by_intent) -> int:
    """Offline: print the deterministic gold and confirm the scenarios create the intended divergences
    (constrained/outage differ from the catalogue oracle), with no API calls."""
    print("E15 deliverability scenarios -- deterministic check\n")
    ok = True
    any_constrained_divergence = False
    for iid in NEG_INTENTS:
        opts = options_by_intent[iid]
        print(f"intent {iid}: options {[o['id'] for o in opts]}")
        for scen, sat in SCENARIOS[iid].items():
            saturated = set(sat)
            line = []
            for pid, pol in policies.items():
                cat = catalogue_offer(intents[iid]["bounds"], opts, pol["priority"])
                g_off, g_dec = deliverability_gold(intents[iid]["bounds"], opts, pol["priority"],
                                                   saturated, pol)
                diverges = g_off != cat
                if scen in ("constrained", "outage") and diverges:
                    any_constrained_divergence = True
                cat_undeliv = cat in saturated
                line.append(f"{pid}: cat={cat}{'(undeliv!)' if cat_undeliv else ''} -> deliv={g_off}/{g_dec}"
                            f"{' [DIVERGE]' if diverges else ''}")
            print(f"  {scen:12} saturated={sat}")
            for l in line:
                print(f"      {l}")
        print()
    # sanity: clear scenario must never diverge (deliverability-best == catalogue-best)
    for iid in NEG_INTENTS:
        opts = options_by_intent[iid]
        for pid, pol in policies.items():
            cat = catalogue_offer(intents[iid]["bounds"], opts, pol["priority"])
            g_off, _ = deliverability_gold(intents[iid]["bounds"], opts, pol["priority"], set(), pol)
            if g_off != cat:
                print(f"  ! clear-scenario divergence at {iid}/{pid}: {g_off} != {cat}")
                ok = False
    print("clear scenario reproduces the catalogue oracle everywhere:", ok)
    print("constrained/outage scenarios produce at least one divergence:", any_constrained_divergence)
    print("OK" if (ok and any_constrained_divergence) else "CHECK FAILED")
    return 0 if (ok and any_constrained_divergence) else 1


if __name__ == "__main__":
    raise SystemExit(main())
