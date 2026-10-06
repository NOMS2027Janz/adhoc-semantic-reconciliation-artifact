#!/usr/bin/env python3
"""Breadth case for the intent setting: an enterprise metro-Ethernet service scenario.

A second, independent instance of the intent mechanism (refinement, negotiation, movable
policy, multi-hop lifecycle), on a different domain (business sites over a metro-Ethernet
carrier) but the same four bound dimensions (bandwidth, latency, availability, protection),
so it reuses reconcile.intent unchanged. Writes benchmark/cases/intent_metro/{intents,
catalogue,policies,lifecycle,intent_traps,intent_reference}.json; the gold is derived and
validated by:  python benchmark/derive_intent_gold.py intent_metro
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "benchmark" / "cases" / "intent_metro"


def A(bw, lat, av, prot, cost):
    return {"bw_gbps": bw, "latency_ms": lat, "availability": av, "protection": prot, "cost": cost}


def B(bw=None, lat=None, av=None, prot=False):
    return {"bandwidth_min": bw, "latency_max": lat, "availability_min": av,
            "protection_required": prot}


# --- realisation catalogue: metro-Ethernet options per site pair -----------------------------
REALISATIONS = [
    # pair M1 (HQ-Branch)
    dict(id="m1a", pair="M1", path="direct", capacity="1GbE", advertised=A(1.0, 8, 0.999, "none", 20)),
    dict(id="m1b", pair="M1", path="protected", capacity="1GbE", advertised=A(1.0, 10, 0.9999, "1+1", 40)),
    # pair M2 (HQ-DataCentre)
    dict(id="m2a", pair="M2", path="direct", capacity="10GbE", advertised=A(10.0, 4, 0.9999, "none", 50)),
    dict(id="m2b", pair="M2", path="protected", capacity="10GbE", advertised=A(10.0, 6, 0.99999, "1+1", 80)),
    dict(id="m2c", pair="M2", path="protected-fast", capacity="10GbE", advertised=A(10.0, 5, 0.99999, "1+1", 95)),
    # pair M3 (Branch-DataCentre) -- m3a deviates: advertised latency 7ms, actual 5ms
    dict(id="m3a", pair="M3", path="direct", capacity="1GbE",
         advertised=A(1.0, 7, 0.999, "none", 18), actual=A(1.0, 5, 0.999, "none", 18)),
    # pair M4 (HQ-Cloud) -- m4b deviates: advertised availability .9999, actual .9995
    dict(id="m4a", pair="M4", path="direct", capacity="2GbE", advertised=A(2.0, 12, 0.999, "none", 30)),
    dict(id="m4b", pair="M4", path="protected", capacity="2GbE",
         advertised=A(2.0, 14, 0.9999, "1+1", 55), actual=A(2.0, 14, 0.9995, "1+1", 55)),
]

INTENTS = [
    dict(id="MI1", pair="M1", flow_class="business-data", bounds=B(0.5, 10, 0.999, False),
         note="fully satisfiable on the direct 1G path"),
    dict(id="MI2", pair="M2", flow_class="critical-txn", bounds=B(10, 6, 0.9999, True),
         note="fully satisfiable on the protected 10G path"),
    dict(id="MI3", pair="M2", flow_class="critical-txn", bounds=B(10, 4.5, 0.99999, True),
         note="infeasible; the accept/reject FLIPS with the policy (premium holds latency and refers; assured degrades latency and accepts)"),
    dict(id="MI4", pair="M1", flow_class="business-data", bounds=B(1, 6, 0.9999, True),
         note="infeasible; only a slower protected path exists, so the decision flips with the policy"),
    dict(id="MI5", pair="M3", flow_class="business-data", bounds=B(1, 6, 0.999, False), experiment_only=True,
         note="EXPERIMENT-ONLY: advertised latency (7ms) says infeasible; the live path is 5ms and actually satisfies - only a feasibility probe reveals it"),
    dict(id="MI6", pair="M4", flow_class="critical-txn", bounds=B(2, 15, 0.9999, True), experiment_only=True,
         note="EXPERIMENT-ONLY: advertised availability (.9999) says satisfiable; the live availability is .9995 and actually breaches - a probe catches the false refine"),
]

POLICIES = [
    dict(id="premium", flow_class="critical-txn",
         priority=["latency", "availability", "protection", "bandwidth"],
         hard_bounds=["latency", "availability"], affordability_floor=90,
         note="latency-first: hold latency and availability, degrade protection/bandwidth if forced"),
    dict(id="assured", flow_class="critical-txn",
         priority=["availability", "protection", "latency", "bandwidth"],
         hard_bounds=["availability", "protection"], affordability_floor=100,
         note="resilience-first: hold availability and protection, tolerate a slower path"),
    dict(id="economy", flow_class="business-data",
         priority=["bandwidth", "latency", "availability", "protection"],
         hard_bounds=["bandwidth"], affordability_floor=40,
         note="cost-sensitive: hold only a bandwidth floor and step aside on a tight budget"),
]

MI1_BOUNDS = B(0.5, 10, 0.999, False)
DEMAND = B(0.5, 8, 0.99999, False)      # T_metro1 h2 consumer demand: five-nines and tighter latency
MI2_BOUNDS = B(10, 6, 0.9999, True)

TRAJECTORIES = [
    dict(id="TM1", intent_id="MI1", policy_id="premium", setpiece=True, initial_realisation="m1a",
         narrative="A metro business-data service across its life: bought, a provider-side latency "
                   "degradation self-remediated by moving to the protected path, a consumer five-nines "
                   "demand that must be referred, and a provider restore to the cheaper direct path once "
                   "the fault clears.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=MI1_BOUNDS,
                  event="service in operation; routine assurance reading"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=MI1_BOUNDS,
                  event="the direct path's latency has degraded past its bound"),
             dict(hop_id="h2", origin="consumer", kind="demand", bounds=DEMAND,
                  event="the consumer now requires five-nines availability for a critical window"),
             dict(hop_id="h3", origin="provider", kind="restore", bounds=MI1_BOUNDS, target="m1a",
                  event="the direct-path fault has cleared; the provider offers to move back to the cheaper direct route"),
         ]),
    dict(id="TM2", intent_id="MI2", policy_id="assured", setpiece=False, initial_realisation="m2b",
         narrative="A resilience-first critical service whose latency drifts past its bound and is "
                   "self-remediated onto the fast protected path.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=MI2_BOUNDS,
                  event="routine assurance; latency sitting near its margin"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=MI2_BOUNDS,
                  event="latency degrades past the agreed bound; the fast protected path can restore it"),
         ]),
]

OPERATIONAL = {
    "TM1/h0": {"realisation": "m1a", "reading": A(1.0, 7, 0.9995, "none", 20)},
    "TM1/h1": {"realisation": "m1a", "reading": A(1.0, 11, 0.9995, "none", 20)},
    "TM1/h2": {"realisation": "m1b", "reading": A(1.0, 10, 0.9999, "1+1", 40)},
    "TM1/h3": {"realisation": "m1b", "reading": A(1.0, 10, 0.9999, "1+1", 40)},
    "TM2/h0": {"realisation": "m2b", "reading": A(10.0, 6, 0.99999, "1+1", 80)},
    "TM2/h1": {"realisation": "m2b", "reading": A(10.0, 7, 0.99999, "1+1", 80)},
}

UNIT_VALUE_SET = [
    {"term": "bandwidth", "kind": "unit",
     "meaning": "an expectation bound in Gbit/s; realised by a metro-Ethernet access rate (1GbE, 10GbE). "
                "The committed rate must meet or exceed the bound."},
    {"term": "latency", "kind": "kind-separation",
     "meaning": "an expectation 'latency <= X ms' is a BOUND, not a measured value; a telemetry "
                "'latency' reading is a metric. Do not conflate the expectation with the metric."},
    {"term": "availability", "kind": "value-set",
     "meaning": "stated in nines; .9999 = four-nines, .99999 = five-nines. A higher requirement is a strictly tighter bound."},
    {"term": "protection", "kind": "value-set",
     "meaning": "protection classes: 'none' | '1+1'. 'protection required' is satisfied only by a protected ('1+1') realisation."},
]


def main() -> int:
    CDIR.mkdir(parents=True, exist_ok=True)
    (CDIR / "intents.json").write_text(json.dumps(
        {"system": "Agent O", "dialect": "TM Forum intent (TMF921 / IG 1253)",
         "intents": [{k: v for k, v in i.items() if k != "experiment_only"} | (
             {"experiment_only": True} if i.get("experiment_only") else {}) for i in INTENTS]},
        indent=2) + "\n")
    (CDIR / "catalogue.json").write_text(json.dumps(
        {"system": "Agent N", "dialect": "MEF metro-Ethernet service catalogue",
         "realisations": [{"id": r["id"], "pair": r["pair"], "path": r["path"],
                           "capacity": r["capacity"], "advertised": r["advertised"]}
                          for r in REALISATIONS]}, indent=2) + "\n")
    (CDIR / "policies.json").write_text(json.dumps({"policies": POLICIES}, indent=2) + "\n")
    (CDIR / "lifecycle.json").write_text(json.dumps({"trajectories": TRAJECTORIES}, indent=2) + "\n")
    (CDIR / "intent_traps.json").write_text(json.dumps({
        "case": "intent_metro",
        "operational_case": "intent (declarative demand -> concrete realisation, by refinement)",
        "seed": "hand-authored breadth case: enterprise metro-Ethernet intents x a priced metro catalogue",
        "margin": 0.10,
        "actual_attrs": {r["id"]: r.get("actual", r["advertised"]) for r in REALISATIONS},
        "operational": OPERATIONAL,
        "experiment_only": [i["id"] for i in INTENTS if i.get("experiment_only")],
        "traps_note": {
            "nature_false_cognate": "an expectation 'latency <= X' vs a measured 'latency' metric must not be conflated",
            "unit_scale": "bandwidth bounds in Gbit/s vs metro access rates must be compared correctly"},
    }, indent=2) + "\n")
    (CDIR / "intent_reference.json").write_text(json.dumps({
        "note": "Two reference arms. 'unit_value_set' pins units, value sets, and the expectation-vs-"
                "realisation kind separation; 'invariant' publishes the committed guarantee floor a "
                "satisfaction check evaluates against when a side is inert.",
        "unit_value_set": UNIT_VALUE_SET,
        "invariant": [{"id": r["id"], "pair": r["pair"], "guaranteed": r["advertised"]} for r in REALISATIONS],
    }, indent=2) + "\n")
    print(f"wrote intent_metro: {len(INTENTS)} intents, {len(REALISATIONS)} realisations, "
          f"{len(POLICIES)} policies, {len(TRAJECTORIES)} trajectories, "
          f"{sum(1 for i in INTENTS if i.get('experiment_only'))} experiment-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
