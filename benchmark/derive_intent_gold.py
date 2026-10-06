#!/usr/bin/env python3
"""Derive and validate benchmark/cases/intent_hard/intent_gold.json from the domain
predicates in reconcile.intent and the hidden operational truth.

    python benchmark/derive_intent_gold.py

Everything scorable is COMPUTED here, so the gold cannot drift from the case:
  * refine-down: the realisations that actually satisfy each intent (by hidden truth),
    and whether advertised evidence agrees (the experiment-only test);
  * negotiation: the best-achievable offer and the accept/reject decision for every
    infeasible intent under every policy;
  * lifecycle: a deterministic walk of each multi-hop trajectory — fulfilment, decision,
    migration, and remediation class per hop, state carried forward.

It refuses to write an inconsistent gold. In particular it proves that every
experiment-only intent genuinely needs the live feasibility check (advertised and actual
disagree), that every infeasible intent truly has no actual satisfier, that the
accept/reject decision actually VARIES with the policy (pragmatic sensitivity is real),
and that the seeded telemetry matches the simulated in-service realisation at each hop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.intent import (                                        # noqa: E402
    BOUND_KINDS, load_intent_case, violated_bounds, satisfies_all,
    best_achievable, policy_decision, fulfilment_status, remediation_class)

CASE = sys.argv[1] if (len(sys.argv) > 1 and not sys.argv[1].startswith("-")) else "intent_hard"
CDIR = ROOT / "benchmark" / "cases" / CASE


def actual_satisfiers(case, intent):
    return [o["id"] for o in case.options_for(intent["pair"], actual=True)
            if satisfies_all(intent["bounds"], o["attrs"])]


def advertised_satisfiers(case, intent):
    return [o["id"] for o in case.options_for(intent["pair"], actual=False)
            if satisfies_all(intent["bounds"], o["attrs"])]


def simulate_trajectory(case, traj, errors):
    """Walk a trajectory deterministically, returning per-hop gold."""
    intent = case.intent_by_id[traj["intent_id"]]
    policy = case.policy_by_id[traj["policy_id"]]
    pair = intent["pair"]
    current = traj["initial_realisation"]
    hops_gold = []
    for hop in traj["hops"]:
        key = f"{traj['id']}/{hop['hop_id']}"
        op = case.traps.get("operational", {}).get(key)
        if not op:
            errors.append(f"{key}: no operational telemetry seeded")
            break
        if op["realisation"] != current:
            errors.append(f"{key}: telemetry realisation '{op['realisation']}' != simulated "
                          f"in-service '{current}'")
        reading = op["reading"]
        bounds = hop["bounds"]
        fulfil = fulfilment_status(bounds, reading, case.traps.get("margin", 0.10))
        if hop["kind"] == "restore":
            target = hop["target"]
            dec = policy_decision(bounds, case.actual_attrs[target], policy)
            decision = "accept" if dec == "accept" else "refer"
            new = target if dec == "accept" else current
            remed = "self_remediable" if dec == "accept" else "needs_consumer"
        elif fulfil == "breach":
            cand = [o for o in case.options_for(pair, actual=True) if o["id"] != current]
            best = best_achievable(bounds, cand, policy["priority"])
            dec = policy_decision(bounds, best["attrs"], policy) if best else "reject"
            decision = "accept" if dec == "accept" else "refer"
            new = best["id"] if (best and dec == "accept") else current
            remed = remediation_class(bounds, cand, policy)
        else:
            decision, new, remed = "hold", current, None
        hops_gold.append({"hop_id": hop["hop_id"], "origin": hop["origin"], "kind": hop["kind"],
                          "fulfilment": fulfil, "decision": decision, "new_realisation": new,
                          "remediation_class": remed, "in_service_before": current})
        current = new
    return hops_gold


def main() -> int:
    case = load_intent_case(CDIR)
    errors: list[str] = []

    # --- policies well-formed --------------------------------------------------------------
    for p in case.policies:
        if set(p["priority"]) != set(BOUND_KINDS):
            errors.append(f"policy '{p['id']}' priority must cover exactly {BOUND_KINDS}")
        if not set(p["hard_bounds"]) <= set(p["priority"]):
            errors.append(f"policy '{p['id']}' hard_bounds not within priority")

    # --- refine-down gold + experiment-only test -------------------------------------------
    refine = {}
    negotiation_intents = []
    for it in case.intents:
        act = actual_satisfiers(case, it)
        adv = advertised_satisfiers(case, it)
        eo = bool(it.get("experiment_only"))
        refine[it["id"]] = {
            "pair": it["pair"], "satisfiable": bool(act), "actual_satisfiers": act,
            "advertised_satisfiers": adv, "experiment_only": eo,
            "correct_satisfies": bool(act)}
        agree = (bool(act) == bool(adv))
        if eo and agree:
            errors.append(f"intent '{it['id']}' flagged experiment-only but advertised and actual "
                          f"agree ({act} / {adv}); it would not need a live probe")
        if not eo and not agree:
            errors.append(f"intent '{it['id']}' NOT experiment-only but advertised/actual disagree "
                          f"({adv} vs {act}); label it experiment-only or fix the case")
        if not act:
            negotiation_intents.append(it["id"])

    # --- negotiation gold: best-achievable + decision per (infeasible intent, policy) -------
    negotiation = {}
    for iid in negotiation_intents:
        it = case.intent_by_id[iid]
        options = case.options_for(it["pair"], actual=True)
        per_policy = {}
        for p in case.policies:
            best = best_achievable(it["bounds"], options, p["priority"])
            if not best:
                errors.append(f"negotiation '{iid}': no options for pair {it['pair']}")
                continue
            dec = policy_decision(it["bounds"], best["attrs"], p)
            per_policy[p["id"]] = {
                "best_offer": best["id"], "decision": dec,
                "offer_violated": sorted(violated_bounds(it["bounds"], best["attrs"])),
                "offer_cost": best["attrs"].get("cost")}
        negotiation[iid] = per_policy

    # pragmatic sensitivity: some intent's decision must vary across policies
    varies = any(len({pp["decision"] for pp in per.values()}) > 1 for per in negotiation.values())
    if negotiation and not varies:
        errors.append("no infeasible intent changes its accept/reject across policies; the "
                      "pragmatic-sensitivity axis would be inert")

    # --- lifecycle gold: deterministic multi-hop walk --------------------------------------
    lifecycle = {}
    setpiece = None
    for traj in case.lifecycle:
        if traj["intent_id"] not in case.intent_by_id:
            errors.append(f"trajectory '{traj['id']}' references unknown intent")
            continue
        if traj["policy_id"] not in case.policy_by_id:
            errors.append(f"trajectory '{traj['id']}' references unknown policy")
            continue
        lifecycle[traj["id"]] = {
            "intent_id": traj["intent_id"], "policy_id": traj["policy_id"],
            "setpiece": bool(traj.get("setpiece")),
            "hops": simulate_trajectory(case, traj, errors)}
        if traj.get("setpiece"):
            setpiece = traj["id"]
    if case.lifecycle and setpiece is None:
        errors.append("no trajectory is flagged as the worked set-piece")

    if errors:
        print("INTENT GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    gold = {
        "case": CASE,
        "operational_case": case.traps.get("operational_case", ""),
        "seed": case.traps.get("seed", ""),
        "note": "DERIVED by derive_intent_gold.py from the domain predicates and hidden truth; "
                "do not hand-edit.",
        "refine": refine,
        "negotiation_intents": negotiation_intents,
        "negotiation": negotiation,
        "lifecycle": lifecycle,
        "setpiece": setpiece,
        "experiment_only": [i for i, r in refine.items() if r["experiment_only"]],
    }
    (CDIR / "intent_gold.json").write_text(json.dumps(gold, indent=2) + "\n")

    print(f"Wrote {CASE}/intent_gold.json")
    print(f"  intents: {len(refine)}  (satisfiable: {sum(1 for r in refine.values() if r['satisfiable'])}, "
          f"experiment-only: {len(gold['experiment_only'])})")
    print(f"  negotiation cases: {len(negotiation_intents)} intents x {len(case.policies)} policies")
    for iid, per in negotiation.items():
        decs = ", ".join(f"{pid}:{pp['decision']}({pp['best_offer']})" for pid, pp in per.items())
        print(f"    {iid}: {decs}")
    print(f"  trajectories: {len(lifecycle)} (set-piece: {setpiece})")
    for tid, tg in lifecycle.items():
        chain = " -> ".join(f"{h['hop_id']}:{h['fulfilment']}/{h['decision']}" for h in tg["hops"])
        print(f"    {tid} [{tg['policy_id']}]: {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
