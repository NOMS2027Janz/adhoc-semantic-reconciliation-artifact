#!/usr/bin/env python3
"""Construct the seeded intent case for the second operational setting.

Writes benchmark/cases/intent_hard/{intents,catalogue,policies,lifecycle,intent_traps,
intent_reference}.json, and a small instance-format endpoint co-reference sub-case at
benchmark/cases/intent_endpoints/ (reused verbatim by the instance agent for phase 3).

The gold is NOT written here; it is derived and validated by derive_intent_gold.py from
the domain predicates in reconcile.intent, so it cannot drift.

Design (mirrors the first setting's discipline):
  * Intents (Agent O) declare quantitative BOUNDS over an endpoint pair, plus a flow class.
    A bound is satisfied by, not equal to, a realisation.
  * The realisation catalogue (Agent N) advertises attributes per option; the oracle holds
    the hidden OPERATIONAL truth (``actual_attrs``), which coincides with advertised except
    on the seeded EXPERIMENT-ONLY intents, where advertised and actual give different
    satisfaction verdicts — resolvable only by a live feasibility check (the analog of the
    instance study's static-twin-needs-a-probe).
  * Movable policies carry the pragmatics: a priority order, the hard (must-hold) bounds,
    an affordability floor, and a flow class. The same infeasible intent decided under
    different policies yields different accept/reject verdicts.
  * Multi-hop lifecycle trajectories run a service across its life: provider- and
    consumer-initiated changes in turn, each a fresh reconciliation, state carried forward.
    One trajectory (T1) is the worked set-piece.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.intent import ODU_BW                                    # noqa: E402

CDIR = ROOT / "benchmark" / "cases" / "intent_hard"
EDIR = ROOT / "benchmark" / "cases" / "intent_endpoints"


def A(bw, lat, av, prot, cost):
    return {"bw_gbps": bw, "latency_ms": lat, "availability": av, "protection": prot, "cost": cost}


# --- realisation catalogue: advertised attrs; actual == advertised unless 'actual' given ----
# Each realisation realises one endpoint PAIR by a path, at an ODU rate.
REALISATIONS = [
    # pair P1 (A1-A3)
    dict(id="r1a", pair="P1", path="direct", capacity="ODU2",
         advertised=A(10.0, 3, 0.99995, "none", 40)),
    dict(id="r1b", pair="P1", path="diverse-protected", capacity="ODU2",
         advertised=A(10.0, 6, 0.99999, "1+1", 70)),
    dict(id="r1c", pair="P1", path="diverse-fast", capacity="ODU2",
         advertised=A(10.0, 4, 0.99999, "1+1", 85)),
    # pair P2 (A2-A3)  -- r2b deviates: advertised availability .9999, actual .9995
    dict(id="r2a", pair="P2", path="direct", capacity="ODU0",
         advertised=A(1.25, 4, 0.999, "none", 20)),
    dict(id="r2b", pair="P2", path="diverse-protected", capacity="ODU1",
         advertised=A(2.5, 7, 0.9999, "1+1", 45),
         actual=A(2.5, 7, 0.9995, "1+1", 45)),
    # pair P3 (A1-A2)
    dict(id="r3a", pair="P3", path="direct", capacity="ODU1",
         advertised=A(2.5, 5, 0.9995, "none", 25)),
    # pair P4 (A1-A4)
    dict(id="r4a", pair="P4", path="direct", capacity="ODU3",
         advertised=A(40.0, 8, 0.999, "none", 60)),
    dict(id="r4b", pair="P4", path="diverse-protected", capacity="ODU3",
         advertised=A(40.0, 12, 0.99995, "1+1", 95)),
    # pair P5 (A2-A4)  -- r5a deviates: advertised latency 9ms, actual 7ms
    dict(id="r5a", pair="P5", path="direct", capacity="ODU2",
         advertised=A(10.0, 9, 0.999, "none", 35),
         actual=A(10.0, 7, 0.999, "none", 35)),
]

# --- intents (Agent O): bounds over a pair, a flow class, and the trap flags -----------------
# bounds: bandwidth_min (Gbit/s), latency_max (ms), availability_min (fraction),
#         protection_required (bool). A None/absent bound is unconstrained.
def B(bw=None, lat=None, av=None, prot=False):
    return {"bandwidth_min": bw, "latency_max": lat, "availability_min": av,
            "protection_required": prot}


INTENTS = [
    dict(id="I1", pair="P1", flow_class="order-execution", bounds=B(8, 5, 0.9999, False),
         note="fully satisfiable on the direct 10G path"),
    dict(id="I2", pair="P1", flow_class="order-execution", bounds=B(8, 3.5, 0.99999, True),
         note="infeasible: no P1 realisation meets sub-3.5ms AND five-nines AND protection"),
    dict(id="I3", pair="P4", flow_class="market-data", bounds=B(40, 10, 0.9999, True),
         note="infeasible; the accept/reject FLIPS with the policy (latency- vs resilience-first)"),
    dict(id="I4", pair="P2", flow_class="bulk", bounds=B(2, 8, 0.999, False),
         note="fully satisfiable on the protected ODU1 path"),
    dict(id="I5", pair="P3", flow_class="order-execution", bounds=B(2, 5, 0.999, False),
         note="fully satisfiable on the direct ODU1 path"),
    dict(id="I6", pair="P5", flow_class="order-execution", bounds=B(8, 8, 0.999, False),
         experiment_only=True,
         note="EXPERIMENT-ONLY: advertised latency (9ms) says infeasible; the live path is 7ms "
              "and actually satisfies — only a feasibility probe reveals it"),
    dict(id="I7", pair="P2", flow_class="order-execution", bounds=B(2, 8, 0.9999, False),
         experiment_only=True,
         note="EXPERIMENT-ONLY: advertised availability (.9999) says satisfiable; the live "
              "availability is .9995 and actually breaches — a probe catches the false refine"),
]

# --- movable policies (the pragmatic artefact) ----------------------------------------------
# priority: bound kinds most-important first. hard_bounds: must-hold. affordability_floor: max cost.
POLICIES = [
    dict(id="exec", flow_class="order-execution",
         priority=["latency", "availability", "protection", "bandwidth"],
         hard_bounds=["latency", "availability"], affordability_floor=90,
         note="latency-first: hold latency and availability, degrade protection/bandwidth if forced"),
    dict(id="resil", flow_class="market-data",
         priority=["availability", "protection", "latency", "bandwidth"],
         hard_bounds=["availability", "protection"], affordability_floor=100,
         note="resilience-first: hold availability and protection, tolerate a slower path"),
    dict(id="bulk", flow_class="bulk",
         priority=["bandwidth", "latency", "availability", "protection"],
         hard_bounds=["bandwidth"], affordability_floor=40,
         note="cost-sensitive: hold only a bandwidth floor and step aside (a tight budget) to save money"),
]

# --- multi-hop lifecycle trajectories -------------------------------------------------------
# Each hop: origin (consumer|provider), kind (assess|demand|restore), the BOUNDS in force at
# that hop (the current agreed SLA — an accepted degraded offer resets the SLA to its terms),
# and, for a 'restore', the offered target. The in-service realisation and its live reading at
# each hop live in intent_traps.operational (hidden telemetry), keyed traj/hop.
T_EXEC_BOUNDS = B(8, 5, 0.9999, False)                 # I1's ask, the agreed SLA on T1
SIXNINES = B(8, 5, 0.999999, False)                    # T1 h2 consumer demand: six-nines availability
RESIL_SLA = B(40, 12, 0.9999, True)                    # T2 agreed SLA after accepting r4b (lat relaxed to 12)
BULK_SLA = B(2, 8, 0.999, False)                       # T3 agreed SLA = I4
BULK_STEPDOWN = B(1, 8, 0.999, False)                  # T3 h1 consumer steps aside to a smaller workload

TRAJECTORIES = [
    dict(id="T1", intent_id="I1", policy_id="exec", setpiece=True, initial_realisation="r1a",
         narrative="An order-execution service across its life: bought, a provider-side latency "
                   "degradation self-remediated by reroute, a consumer six-nines demand that must "
                   "be referred, and a provider restore to the cheaper path once the fault clears.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=T_EXEC_BOUNDS,
                  event="service in operation; routine assurance reading"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=T_EXEC_BOUNDS,
                  event="the direct path's latency has degraded; assurance predicts a breach"),
             dict(hop_id="h2", origin="consumer", kind="demand", bounds=SIXNINES,
                  event="the consumer now requires six-nines availability for a critical window"),
             dict(hop_id="h3", origin="provider", kind="restore", bounds=T_EXEC_BOUNDS,
                  target="r1a",
                  event="the direct-path fault has cleared; the provider offers to move back to the "
                        "cheaper direct route"),
         ]),
    dict(id="T2", intent_id="I3", policy_id="resil", setpiece=False, initial_realisation="r4b",
         narrative="A resilience-first service: an at-risk latency margin held, then a degradation "
                   "the network cannot self-remediate without losing protection — referred.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=RESIL_SLA,
                  event="routine assurance; latency sitting near its margin"),
             dict(hop_id="h1", origin="provider", kind="assess", bounds=RESIL_SLA,
                  event="latency degrades past the agreed bound; only an unprotected alternative exists"),
         ]),
    dict(id="T3", intent_id="I4", policy_id="bulk", setpiece=False, initial_realisation="r2b",
         narrative="A bulk service that steps aside to save money when the workload shrinks.",
         hops=[
             dict(hop_id="h0", origin="provider", kind="assess", bounds=BULK_SLA,
                  event="routine assurance; comfortably met"),
             dict(hop_id="h1", origin="consumer", kind="restore", bounds=BULK_STEPDOWN,
                  target="r2a",
                  event="the workload has shrunk; the consumer steps aside to the cheaper direct path"),
         ]),
]

# hidden operational telemetry per (trajectory, hop): the in-service realisation and its live reading
OPERATIONAL = {
    "T1/h0": {"realisation": "r1a", "reading": A(10.0, 3, 0.99995, "none", 40)},
    "T1/h1": {"realisation": "r1a", "reading": A(10.0, 6, 0.99995, "none", 40)},   # latency degraded
    "T1/h2": {"realisation": "r1c", "reading": A(10.0, 4, 0.99999, "1+1", 85)},    # now on r1c after h1
    "T1/h3": {"realisation": "r1c", "reading": A(10.0, 4, 0.99999, "1+1", 85)},    # still r1c; restore offers r1a
    "T2/h0": {"realisation": "r4b", "reading": A(40.0, 12, 0.99995, "1+1", 95)},   # at margin
    "T2/h1": {"realisation": "r4b", "reading": A(40.0, 14, 0.99995, "1+1", 95)},   # latency past bound
    "T3/h0": {"realisation": "r2b", "reading": A(2.5, 7, 0.9995, "1+1", 45)},
    "T3/h1": {"realisation": "r2b", "reading": A(2.5, 7, 0.9995, "1+1", 45)},      # fine; step-down offers r2a
}

# --- references (two arms) ------------------------------------------------------------------
UNIT_VALUE_SET = [
    {"term": "bandwidth", "kind": "unit",
     "meaning": "an expectation bound in Gbit/s; realised by a discrete ODU client signal. "
                "ODU rates map to usable bandwidth: " + ", ".join(f"{k}~{v}Gbit/s" for k, v in ODU_BW.items())},
    {"term": "latency", "kind": "kind-separation",
     "meaning": "an expectation 'latency <= X ms' is a BOUND, not a measured value; a telemetry "
                "'latency' reading is a metric. Do not conflate the expectation with the metric."},
    {"term": "availability", "kind": "value-set",
     "meaning": "stated in nines; .9999 = four-nines, .99999 = five-nines. A higher requirement is "
                "a strictly tighter bound."},
    {"term": "protection", "kind": "value-set",
     "meaning": "protection classes: 'none' | '1+1'. 'protection required' is satisfied only by a "
                "protected ('1+1') realisation."},
]


def build_catalogue_rows():
    rows = []
    for r in REALISATIONS:
        rows.append({"id": r["id"], "pair": r["pair"], "path": r["path"],
                     "capacity": r["capacity"], "advertised": r["advertised"]})
    return rows


def build_actual_attrs():
    return {r["id"]: r.get("actual", r["advertised"]) for r in REALISATIONS}


def build_invariant_reference():
    # publishes the COMMITTED (advertised) guarantee floor per realisation — an anchor a
    # satisfaction check can evaluate against when a side is inert. It does NOT reveal the
    # hidden operational deviations (those stay probe-only), preserving the experiment-only trap.
    return [{"id": r["id"], "pair": r["pair"], "guaranteed": r["advertised"]} for r in REALISATIONS]


# --- endpoint co-reference sub-case (instance format, reused by the instance agent) ---------
def build_endpoint_case():
    """A small A-box co-reference problem: service-order keys (Agent O) <-> UNI ids (Agent N)
    <-> location. Includes one experiment-only pair separable only by an interrogated fibre-id.
    Emitted in the instance-study format with a self-derived gold."""
    ACCESS = "client-svc-access"
    ents = [
        dict(truth="u1", type=ACCESS, key="so-501",
             a=dict(id="o.ep1", name="EP-A1", attrs={"site": "NY-DC1"}),
             b=dict(id="n.uni1", name="uni-1", attrs={"site": "NY-DC1"})),
        dict(truth="u2", type=ACCESS, key="so-502",
             a=dict(id="o.ep2", name="EP-A2", attrs={"site": "CHI-DC1"}),
             b=dict(id="n.uni2", name="uni-2", attrs={"site": "CHI-DC1"})),
        dict(truth="u3", type=ACCESS, key="so-503",
             a=dict(id="o.ep3", name="EP-A3", attrs={"site": "FRA-DC1"}),
             b=dict(id="n.uni3", name="uni-3", attrs={"site": "FRA-DC1"})),
        # experiment-only twins: same site, no key; separable only by an interrogated fibre-id
        dict(truth="u4", type=ACCESS, experiment_only=True,
             a=dict(id="o.ep4", name="EP-A4", attrs={"site": "LDN-DC2"}),
             b=dict(id="n.uni4", name="uni-4", attrs={"site": "LDN-DC2"})),
        dict(truth="u5", type=ACCESS, experiment_only=True,
             a=dict(id="o.ep5", name="EP-A5", attrs={"site": "LDN-DC2"}),
             b=dict(id="n.uni5", name="uni-5", attrs={"site": "LDN-DC2"})),
    ]

    def side_rows(side):
        rows = []
        for e in ents:
            s = e.get(side)
            if not s:
                continue
            rows.append({"id": s["id"], "type": e["type"], "name": s["name"],
                         "key": e.get("key"), "attrs": s.get("attrs", {}), "rels": [],
                         "_truth": e["truth"]})
        return rows

    a_rows, b_rows = side_rows("a"), side_rows("b")
    a_of = {e["truth"]: e["a"]["id"] for e in ents if e.get("a")}
    b_of = {e["truth"]: e["b"]["id"] for e in ents if e.get("b")}
    shared = sorted(set(a_of) & set(b_of))
    correspondences = [{"a": a_of[t], "b": b_of[t], "truth": t} for t in shared]
    experiment_only = [e["truth"] for e in ents if e.get("experiment_only")]
    interr = {"o.ep4": {"fibre_id": "FIB-L4"}, "n.uni4": {"fibre_id": "FIB-L4"},
              "o.ep5": {"fibre_id": "FIB-L5"}, "n.uni5": {"fibre_id": "FIB-L5"}}

    EDIR.mkdir(parents=True, exist_ok=True)
    (EDIR / "individuals_a.json").write_text(json.dumps(
        {"system": "Agent O", "dialect": "TMF service order (endpoints)", "individuals": a_rows}, indent=2) + "\n")
    (EDIR / "individuals_b.json").write_text(json.dumps(
        {"system": "Agent N", "dialect": "IETF L1CSM (UNIs)", "individuals": b_rows}, indent=2) + "\n")
    traps = {
        "case": "intent_endpoints", "operational_case": "intent (endpoint co-reference)",
        "seed": "hand-authored endpoint co-reference sub-case for the intent setting",
        "false_cognates": [], "experiment_only": experiment_only,
        "oracle": {"interrogate": interr, "invariants": {}},
        "residual_by_placement": {"both_cognitive": 0, "one_inert": 2, "both_inert": 2},
        "invariants": [],
        "verification_by_placement": {
            "both_cognitive": "interrogation of the authoritative fibre-id",
            "one_inert": "interrogation on the live side only", "both_inert": "static evidence only"},
    }
    (EDIR / "instance_traps.json").write_text(json.dumps(traps, indent=2) + "\n")
    gold = {
        "case": "intent_endpoints", "note": "DERIVED inline by build_intent_case.py from _truth.",
        "correspondences": correspondences, "false_cognates": [],
        "residual": {"a_only": [], "b_only": []},
        "experiment_only": experiment_only,
        "residual_by_placement": traps["residual_by_placement"],
        "invariants": [], "invariant_signatures": {},
        "verification_by_placement": traps["verification_by_placement"],
    }
    (EDIR / "instance_gold.json").write_text(json.dumps(gold, indent=2) + "\n")
    inst_ref = [{"key": e["key"], "type": e["type"], "canonical": e["a"]["name"]}
                for e in ents if e.get("key")]
    (EDIR / "instance_reference.json").write_text(json.dumps(
        {"note": "opaque service-order keys binding endpoints across O and N.",
         "instance": inst_ref, "invariant": []}, indent=2) + "\n")
    return len(a_rows), len(b_rows), len(correspondences), len(experiment_only)


def main() -> int:
    CDIR.mkdir(parents=True, exist_ok=True)

    (CDIR / "intents.json").write_text(json.dumps(
        {"system": "Agent O", "dialect": "TM Forum intent (TMF921 / IG 1253)",
         "intents": [{k: v for k, v in i.items() if k != "experiment_only"} | (
             {"experiment_only": True} if i.get("experiment_only") else {}) for i in INTENTS]},
        indent=2) + "\n")
    (CDIR / "catalogue.json").write_text(json.dumps(
        {"system": "Agent N", "dialect": "IETF L1CSM / TE service mapping (OTN/optical)",
         "realisations": build_catalogue_rows()}, indent=2) + "\n")
    (CDIR / "policies.json").write_text(json.dumps({"policies": POLICIES}, indent=2) + "\n")
    (CDIR / "lifecycle.json").write_text(json.dumps({"trajectories": TRAJECTORIES}, indent=2) + "\n")

    traps = {
        "case": "intent_hard",
        "operational_case": "intent (declarative demand -> concrete realisation, by refinement)",
        "seed": "hand-authored intents x priced OTN realisation catalogue over the first case's network",
        "margin": 0.10,
        "actual_attrs": build_actual_attrs(),
        "operational": OPERATIONAL,
        "experiment_only": [i["id"] for i in INTENTS if i.get("experiment_only")],
        "traps_note": {
            "nature_false_cognate": "an expectation 'latency <= X' vs a measured 'latency' metric "
                                    "(refine-down vs assure-up) must not be conflated",
            "unit_scale": "bandwidth bounds in Gbit/s vs discrete ODU client rates must be scaled correctly"},
    }
    (CDIR / "intent_traps.json").write_text(json.dumps(traps, indent=2) + "\n")

    (CDIR / "intent_reference.json").write_text(json.dumps(
        {"note": "Two reference arms (design sec. 4). 'unit_value_set' pins units, value sets, and "
                 "the expectation-vs-realisation kind separation; 'invariant' publishes the committed "
                 "guarantee floor a satisfaction check evaluates against when a side is inert.",
         "unit_value_set": UNIT_VALUE_SET, "invariant": build_invariant_reference()}, indent=2) + "\n")

    ea, eb, ecorr, eeo = build_endpoint_case()

    print("wrote intent_hard:",
          f"{len(INTENTS)} intents, {len(REALISATIONS)} realisations, {len(POLICIES)} policies,",
          f"{len(TRAJECTORIES)} trajectories ({sum(len(t['hops']) for t in TRAJECTORIES)} hops),",
          f"{len(traps['experiment_only'])} experiment-only intents")
    print("wrote intent_endpoints:",
          f"A={ea} B={eb} individuals, {ecorr} correspondences, {eeo} experiment-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
