#!/usr/bin/env python3
"""Breadth case for the intent setting: a data-centre-interconnect (DCI) service scenario.

A third, independent instance of the intent mechanism, on a different domain (high-bandwidth
links between data centres) but the same four bound dimensions, reusing reconcile.intent
unchanged. Writes benchmark/cases/intent_dci/{intents,catalogue,policies,lifecycle,
intent_traps,intent_reference}.json; the gold is derived and validated by:
    python benchmark/derive_intent_gold.py intent_dci
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "benchmark" / "cases" / "intent_dci"


def A(bw, lat, av, prot, cost):
    return {"bw_gbps": bw, "latency_ms": lat, "availability": av, "protection": prot, "cost": cost}


def B(bw=None, lat=None, av=None, prot=False):
    return {"bandwidth_min": bw, "latency_max": lat, "availability_min": av,
            "protection_required": prot}


REALISATIONS = [
    # pair D1 (DC-East - DC-West)
    dict(id="d1a", pair="D1", path="direct", capacity="100G", advertised=A(100.0, 20, 0.9999, "none", 200)),
    dict(id="d1b", pair="D1", path="protected", capacity="100G", advertised=A(100.0, 25, 0.99999, "1+1", 320)),
    # pair D2 (DC-East - DC-Central)
    dict(id="d2a", pair="D2", path="direct", capacity="40G", advertised=A(40.0, 10, 0.9999, "none", 120)),
    dict(id="d2b", pair="D2", path="protected", capacity="40G", advertised=A(40.0, 14, 0.99999, "1+1", 190)),
    dict(id="d2c", pair="D2", path="protected-fast", capacity="40G", advertised=A(40.0, 12, 0.99999, "1+1", 230)),
    # pair D3 (DC-West - DC-Central) -- d3a deviates: advertised latency 9ms, actual 6ms
    dict(id="d3a", pair="D3", path="direct", capacity="10G",
         advertised=A(10.0, 9, 0.999, "none", 60), actual=A(10.0, 6, 0.999, "none", 60)),
    # pair D4 (DC-East - Cloud) -- d4b deviates: advertised availability .9999, actual .9994
    dict(id="d4a", pair="D4", path="direct", capacity="25G", advertised=A(25.0, 30, 0.999, "none", 90)),
    dict(id="d4b", pair="D4", path="protected", capacity="25G",
         advertised=A(25.0, 35, 0.9999, "1+1", 150), actual=A(25.0, 35, 0.9994, "1+1", 150)),
]

INTENTS = [
    dict(id="DI1", pair="D1", flow_class="vm-migration", bounds=B(50, 25, 0.9999, False),
         note="fully satisfiable on the direct 100G path"),
    dict(id="DI2", pair="D2", flow_class="storage-sync", bounds=B(40, 14, 0.99999, True),
         note="fully satisfiable on the protected 40G path"),
    dict(id="DI3", pair="D2", flow_class="storage-sync", bounds=B(40, 9, 0.99999, True),
         note="infeasible; the accept/reject FLIPS with the policy (premium holds latency and refers; assured degrades latency and accepts)"),
    dict(id="DI4", pair="D1", flow_class="vm-migration", bounds=B(50, 20, 0.99999, True),
         note="infeasible; only a slower protected path exists, so the decision flips with the policy"),
    dict(id="DI5", pair="D3", flow_class="backup", bounds=B(10, 8, 0.999, False), experiment_only=True,
         note="EXPERIMENT-ONLY: advertised latency (9ms) says infeasible; the live path is 6ms and actually satisfies"),
    dict(id="DI6", pair="D4", flow_class="vm-migration", bounds=B(25, 40, 0.9999, True), experiment_only=True,
         note="EXPERIMENT-ONLY: advertised availability (.9999) says satisfiable; the live availability is .9994 and actually breaches"),
]

POLICIES = [
    dict(id="premium", flow_class="storage-sync",
         priority=["latency", "availability", "protection", "bandwidth"],
         hard_bounds=["latency", "availability"], affordability_floor=350,
         note="latency-first: hold latency and availability, degrade protection/bandwidth if forced"),
    dict(id="assured", flow_class="storage-sync",
         priority=["availability", "protection", "latency", "bandwidth"],
         hard_bounds=["availability", "protection"], affordability_floor=400,
         note="resilience-first: hold availability and protection, tolerate a slower path"),
    dict(id="economy", flow_class="backup",
         priority=["bandwidth", "latency", "availability", "protection"],
         hard_bounds=["bandwidth"], affordability_floor=130,
         note="cost-sensitive: hold only a bandwidth floor and step aside on a tight budget"),
]

DI1_BOUNDS = B(50, 25, 0.9999, False)
DEMAND = B(50, 22, 0.99999, False)
DI2_BOUNDS = B(40, 14, 0.99999, True)

TRAJECTORIES = [
    dict(id="TD1", intent_id="DI1", policy_id="premium", setpiece=True, initial_realisation="d1a",
         narrative="A VM-migration interconnect across its life: bought, a provider latency degradation "
                   "self-remediated onto the protected path, a consumer five-nines demand that must be "
                   "referred, and a provider restore to the cheaper direct path once the fault clears.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=DI1_BOUNDS,
                  event="service in operation; routine assurance reading"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=DI1_BOUNDS,
                  event="the direct path's latency has degraded past its bound"),
             dict(hop_id="h2", origin="consumer", kind="demand", bounds=DEMAND,
                  event="the consumer now requires five-nines availability for a migration window"),
             dict(hop_id="h3", origin="provider", kind="restore", bounds=DI1_BOUNDS, target="d1a",
                  event="the fault has cleared; the provider offers to move back to the cheaper direct route"),
         ]),
    dict(id="TD2", intent_id="DI2", policy_id="assured", setpiece=False, initial_realisation="d2b",
         narrative="A storage-sync interconnect whose latency drifts past its bound and is self-remediated "
                   "onto the fast protected path.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=DI2_BOUNDS,
                  event="routine assurance; latency near its margin"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=DI2_BOUNDS,
                  event="latency degrades past the agreed bound; the fast protected path can restore it"),
         ]),
]

OPERATIONAL = {
    "TD1/h0": {"realisation": "d1a", "reading": A(100.0, 18, 0.99995, "none", 200)},
    "TD1/h1": {"realisation": "d1a", "reading": A(100.0, 27, 0.99995, "none", 200)},
    "TD1/h2": {"realisation": "d1b", "reading": A(100.0, 25, 0.99999, "1+1", 320)},
    "TD1/h3": {"realisation": "d1b", "reading": A(100.0, 25, 0.99999, "1+1", 320)},
    "TD2/h0": {"realisation": "d2b", "reading": A(40.0, 13, 0.99999, "1+1", 190)},
    "TD2/h1": {"realisation": "d2b", "reading": A(40.0, 15, 0.99999, "1+1", 190)},
}

UNIT_VALUE_SET = [
    {"term": "bandwidth", "kind": "unit",
     "meaning": "an expectation bound in Gbit/s; realised by a DCI link rate (10G, 25G, 40G, 100G). "
                "The committed rate must meet or exceed the bound."},
    {"term": "latency", "kind": "kind-separation",
     "meaning": "an expectation 'latency <= X ms' is a BOUND, not a measured value; a telemetry 'latency' "
                "reading is a metric. Do not conflate the expectation with the metric."},
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
        {"system": "Agent N", "dialect": "DCI transport service catalogue",
         "realisations": [{"id": r["id"], "pair": r["pair"], "path": r["path"],
                           "capacity": r["capacity"], "advertised": r["advertised"]}
                          for r in REALISATIONS]}, indent=2) + "\n")
    (CDIR / "policies.json").write_text(json.dumps({"policies": POLICIES}, indent=2) + "\n")
    (CDIR / "lifecycle.json").write_text(json.dumps({"trajectories": TRAJECTORIES}, indent=2) + "\n")
    (CDIR / "intent_traps.json").write_text(json.dumps({
        "case": "intent_dci",
        "operational_case": "intent (declarative demand -> concrete realisation, by refinement)",
        "seed": "hand-authored breadth case: data-centre-interconnect intents x a priced DCI catalogue",
        "margin": 0.10,
        "actual_attrs": {r["id"]: r.get("actual", r["advertised"]) for r in REALISATIONS},
        "operational": OPERATIONAL,
        "experiment_only": [i["id"] for i in INTENTS if i.get("experiment_only")],
        "traps_note": {
            "nature_false_cognate": "an expectation 'latency <= X' vs a measured 'latency' metric must not be conflated",
            "unit_scale": "bandwidth bounds in Gbit/s vs DCI link rates must be compared correctly"},
    }, indent=2) + "\n")
    (CDIR / "intent_reference.json").write_text(json.dumps({
        "note": "Two reference arms. 'unit_value_set' pins units, value sets, and the expectation-vs-"
                "realisation kind separation; 'invariant' publishes the committed guarantee floor.",
        "unit_value_set": UNIT_VALUE_SET,
        "invariant": [{"id": r["id"], "pair": r["pair"], "guaranteed": r["advertised"]} for r in REALISATIONS],
    }, indent=2) + "\n")
    print(f"wrote intent_dci: {len(INTENTS)} intents, {len(REALISATIONS)} realisations, "
          f"{len(POLICIES)} policies, {len(TRAJECTORIES)} trajectories, "
          f"{sum(1 for i in INTENTS if i.get('experiment_only'))} experiment-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
