#!/usr/bin/env python3
"""Construct the seeded hard instance case (the A-boxes, traps, references).

Writes benchmark/cases/instance_hard/{individuals_a,individuals_b,instance_traps,
instance_reference}.json. The gold is NOT written here; it is derived and validated
by derive_instance_gold.py from each individual's hidden `_truth` entity id, so it
cannot drift and the opaque key stays an ablatable piece of evidence rather than the
ground truth.

Design (mirrors the schema benchmark's discipline):
  * Every individual carries a hidden `_truth` (its real-world entity). Two individuals,
    one per graph, with the same `_truth` are the same entity -> a correspondence.
    `_truth` is stripped from anything the agent sees; only the gold reads it.
  * `key` is an OPAQUE shared identifier (k01, ...), present only where an author would
    have one; it coincides with truth where present, and is absent by construction on the
    experiment-only and keyless-ambiguous entities. It is an evidence factor, ablatable.
  * `name` is the local label (the trap surface); `attrs` the attribute vector; `rels`
    the topology to other individuals in the same graph (targets are same-graph ids).
  * Five trap classes are planted: merge targets, instance false cognates, structurally
    symmetric (experiment-only) pairs, native gaps, keyless-ambiguous.
  * Oracle ground truth (interrogation answers; per-correspondence invariant signatures)
    lives in instance_traps.json, never in the individuals, so a static agent cannot read
    what only a live probe or virtual manipulation should reveal.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "benchmark" / "cases" / "instance_hard"

# reconciled schema types (reused from the first case's reference vocabulary)
NODE = "forwarding-node"
ACCESS = "client-svc-access"
SVC = "connection-service"
OMS = "optical-mux-section"

# --- the roster: one dict per truth entity, with optional per-side presentation ----------
# a/b: {id, name, key(optional), attrs(optional), rels(optional list of (rel, truth_target))}
# role: authoring note; experiment_only / keyless flags drive validation and the gold.
ENTITIES = [
    # --- easy merge targets: resolvable statically by key and/or attributes ---
    dict(truth="e1", type=NODE, role="merge-easy",
         a=dict(id="a.r1", name="R1", key="k01", attrs={"role": "core", "model": "MX9"},
                rels=[("adjacent", "e2"), ("adjacent", "e3"), ("adjacent", "e4")]),
         b=dict(id="b.roadm1", name="roadm-1", key="k01", attrs={"role": "core", "model": "MX9"},
                rels=[("adjacent", "e2"), ("adjacent", "e3"), ("adjacent", "e4")])),
    dict(truth="e4", type=NODE, role="merge-easy",
         a=dict(id="a.r4", name="R4", key="k04", attrs={"role": "spur-agg", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e2"), ("adjacent", "e3"), ("adjacent", "e7")]),
         b=dict(id="b.roadm4", name="roadm-4", key="k04", attrs={"role": "spur-agg", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e2"), ("adjacent", "e3"), ("adjacent", "e7")])),
    dict(truth="e5", type=ACCESS, role="merge-easy",
         a=dict(id="a.a1", name="A1", key="k05", attrs={"tier": "access"}, rels=[("homed-on", "e1")]),
         b=dict(id="b.acc1", name="acc-1", key="k05", attrs={"tier": "access"}, rels=[("homed-on", "e1")])),
    dict(truth="e6", type=ACCESS, role="merge-easy",
         a=dict(id="a.a2", name="A2", key="k06", attrs={"tier": "access"}, rels=[("homed-on", "e4")]),
         b=dict(id="b.acc2", name="acc-2", key="k06", attrs={"tier": "access"}, rels=[("homed-on", "e4")])),
    dict(truth="e7", type=ACCESS, role="merge-easy",
         a=dict(id="a.a3", name="A3", key="k07", attrs={"tier": "access"}, rels=[("homed-on", "e4")]),
         b=dict(id="b.acc3", name="acc-3", key="k07", attrs={"tier": "access"}, rels=[("homed-on", "e4")])),

    # --- structurally symmetric pair: no key, identical attrs AND identical neighbour set,
    #     so static evidence (name/attrs/topology) cannot tell which A maps to which B.
    #     Resolvable only by interrogating an authoritative serial (the oracle). experiment-only.
    dict(truth="e2", type=NODE, role="symmetric", experiment_only=True,
         a=dict(id="a.r2", name="R2", attrs={"role": "core", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e4")]),
         b=dict(id="b.nodeB", name="nodeB", attrs={"role": "core", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e4")])),
    dict(truth="e3", type=NODE, role="symmetric", experiment_only=True,
         a=dict(id="a.r3", name="R3", attrs={"role": "core", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e4")]),
         b=dict(id="b.nodeC", name="nodeC", attrs={"role": "core", "model": "MX7"},
                rels=[("adjacent", "e1"), ("adjacent", "e4")])),

    # --- services with invariant signatures: verifiable by virtual manipulation ---
    dict(truth="e8", type=SVC, role="merge-service",
         invariants={"endpoint-identity": ["A1", "A3"], "capacity": "ODU2",
                     "layer-relationships": "ODU2/OTU4", "odu-multiplexing": "ODU2->OTU4",
                     "switching-constraints": "core-transit", "protection-integrity": "1+1"},
         a=dict(id="a.cs_odu2", name="cs-a1a3-odu2", key="k08", attrs={"rate": "ODU2"},
                rels=[("from", "e5"), ("to", "e7")]),
         b=dict(id="b.tunnel1", name="tunnel-1", key="k08", attrs={"rate": "ODU2"},
                rels=[("from", "e5"), ("to", "e7")])),
    dict(truth="e9", type=SVC, role="merge-service",
         invariants={"endpoint-identity": ["A2", "A3"], "capacity": "ODU0",
                     "layer-relationships": "ODU0/ODU2", "odu-multiplexing": "ODU0->ODU2",
                     "switching-constraints": "core-transit", "protection-integrity": "none"},
         a=dict(id="a.cs_odu0", name="cs-a2a3-odu0", key="k09", attrs={"rate": "ODU0"},
                rels=[("from", "e6"), ("to", "e7")]),
         b=dict(id="b.tunnel2", name="tunnel-2", key="k09", attrs={"rate": "ODU0"},
                rels=[("from", "e6"), ("to", "e7")])),

    # --- keyless-ambiguous pair: no key; static attrs underdetermine against a distractor;
    #     resolvable by interrogating an authoritative fibre-id (the oracle). experiment-only. ---
    dict(truth="e13", type=OMS, role="keyless", experiment_only=True,
         a=dict(id="a.omsx", name="oms-x", attrs={"span": "short"},
                rels=[("between", "e1"), ("between", "e4")]),
         b=dict(id="b.msx", name="mux-sec-x", attrs={"span": "short"},
                rels=[("between", "e1"), ("between", "e4")])),
    dict(truth="e14", type=OMS, role="keyless", experiment_only=True,
         a=dict(id="a.omsy", name="oms-y", attrs={"span": "short"},
                rels=[("between", "e1"), ("between", "e4")]),
         b=dict(id="b.msy", name="mux-sec-y", attrs={"span": "short"},
                rels=[("between", "e1"), ("between", "e4")])),

    # --- native gaps that also form instance false cognates on a shared local name ---
    # svc-100 exists on BOTH sides as different real services (different endpoints/capacity):
    # must NOT be merged; a virtual manipulation refutes the merge on the capacity/endpoint invariant.
    dict(truth="e10", type=SVC, role="native-gap-a fc-name",
         invariants={"endpoint-identity": ["A1", "A2"], "capacity": "ODU1"},
         a=dict(id="a.svc100", name="svc-100", attrs={"rate": "ODU1"},
                rels=[("from", "e5"), ("to", "e6")])),
    dict(truth="e11", type=SVC, role="native-gap-b fc-name",
         invariants={"endpoint-identity": ["A3", "A1"], "capacity": "ODU0"},
         b=dict(id="b.svc100", name="svc-100", attrs={"rate": "ODU0"},
                rels=[("from", "e7"), ("to", "e5")])),
    # a second "R1" device on the B side (different real device): the name-collision trap
    # against A's R1 (e1). B-only native gap; correct partner of A's R1 is roadm-1 (e1).
    dict(truth="e12", type=NODE, role="native-gap-b fc-name",
         b=dict(id="b.r1", name="R1", attrs={"role": "core", "model": "MX9"},
                rels=[("adjacent", "e4")])),
]

# --- oracle ground truth ------------------------------------------------------------------
# interrogation: one authoritative fact per individual, revealed only by a live probe.
# The serials separate the symmetric nodes; the fibre-ids separate the keyless OMS pair.
INTERROGATE = {
    "a.r2": {"serial": "SN-0022"}, "b.nodeB": {"serial": "SN-0022"},
    "a.r3": {"serial": "SN-0033"}, "b.nodeC": {"serial": "SN-0033"},
    "a.omsx": {"fibre_id": "FIB-X7"}, "b.msx": {"fibre_id": "FIB-X7"},
    "a.omsy": {"fibre_id": "FIB-Y9"}, "b.msy": {"fibre_id": "FIB-Y9"},
}

FALSE_COGNATES = [
    {"a": "a.r1", "b": "b.r1",
     "why": "both locally named 'R1', but A's R1 is device e1 (its true partner is roadm-1); "
            "B's R1 is a different physical device (e12, a native gap)."},
    {"a": "a.svc100", "b": "b.svc100",
     "why": "both locally named 'svc-100', but different services: A1-A2 ODU1 vs A3-A1 ODU0; "
            "a virtual provision refutes the merge on the endpoint and capacity invariants."},
]

INVARIANTS = ["endpoint-identity", "connectivity", "capacity", "layer-relationships",
              "switching-constraints", "odu-multiplexing", "protection-integrity"]

VERIFICATION_BY_PLACEMENT = {
    "both_cognitive": "interrogation and virtual provision-and-read-back on either side",
    "one_inert": "interrogation/manipulation on the live side; invariant reference on the inert side",
    "both_inert": "static evidence only; propose for external adjudication",
}


def _agent_view(ind: dict) -> dict:
    """The individual as the agent may see it (no hidden truth). Field masking for the
    evidence-ablation factorial is applied later in the driver; this is the full record."""
    out = {"id": ind["id"], "name": ind["name"]}
    if ind.get("key"):
        out["key"] = ind["key"]
    if ind.get("attrs"):
        out["attrs"] = ind["attrs"]
    if ind.get("rels"):
        out["rels"] = ind["rels"]
    return out


def build_side(side: str) -> list[dict]:
    rows = []
    for e in ENTITIES:
        s = e.get(side)
        if not s:
            continue
        rels = [{"rel": r, "target": _side_id(t, side)} for (r, t) in s.get("rels", [])]
        rows.append({
            "id": s["id"], "type": e["type"], "name": s["name"],
            "key": s.get("key"), "attrs": s.get("attrs", {}), "rels": rels,
            "_truth": e["truth"],
        })
    return rows


def _side_id(truth: str, side: str) -> str:
    for e in ENTITIES:
        if e["truth"] == truth and e.get(side):
            return e[side]["id"]
    return f"<{truth}:absent>"  # a rel to an entity not present on this side (rare; flagged)


def main() -> int:
    CDIR.mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = build_side("a"), build_side("b")

    (CDIR / "individuals_a.json").write_text(json.dumps(
        {"system": "Agent T", "dialect": "ONF TAPI (populated)", "individuals": a_rows}, indent=2) + "\n")
    (CDIR / "individuals_b.json").write_text(json.dumps(
        {"system": "Agent I", "dialect": "IETF TEAS (populated)", "individuals": b_rows}, indent=2) + "\n")

    invariants_by_truth = {e["truth"]: e["invariants"] for e in ENTITIES if e.get("invariants")}
    experiment_only = [e["truth"] for e in ENTITIES if e.get("experiment_only")]

    traps = {
        "case": "instance_hard",
        "operational_case": "configuration (instance level)",
        "seed": "hand-authored populated OTN-over-optical A-box over the reconciled TAPI/TEAS types",
        "false_cognates": FALSE_COGNATES,
        "experiment_only": experiment_only,
        "oracle": {"interrogate": INTERROGATE, "invariants": invariants_by_truth},
        "residual_by_placement": {"both_cognitive": 3, "one_inert": 3, "both_inert": 7},
        "invariants": INVARIANTS,
        "verification_by_placement": VERIFICATION_BY_PLACEMENT,
    }
    (CDIR / "instance_traps.json").write_text(json.dumps(traps, indent=2) + "\n")

    # instance reference (opaque keys + canonical descriptor) and invariant reference
    # (published invariant signature), per entity that has a published anchor.
    inst_entries, inv_entries = [], []
    for e in ENTITIES:
        pres = e.get("a") or e.get("b")
        key = (e.get("a") or {}).get("key") or (e.get("b") or {}).get("key")
        if key:
            inst_entries.append({"key": key, "type": e["type"],
                                 "canonical": pres["name"]})
            # the invariant reference publishes signatures ONLY under a published key
            # (never the hidden truth), so it covers keyed services and no native gap.
            if e.get("invariants"):
                inv_entries.append({"key": key, "type": e["type"],
                                    "canonical": pres["name"], "invariants": e["invariants"]})
    (CDIR / "instance_reference.json").write_text(json.dumps(
        {"note": "Two reference variants for the reference-variant axis (see the design note "
                 "sec. 8). 'instance' publishes opaque keys + a canonical descriptor; 'invariant' "
                 "publishes the semantic-invariant signature a virtual manipulation checks against.",
         "instance": inst_entries, "invariant": inv_entries}, indent=2) + "\n")

    print("wrote instance_hard:",
          f"A={len(a_rows)} individuals, B={len(b_rows)} individuals,",
          f"{len(FALSE_COGNATES)} false cognates, {len(experiment_only)} experiment-only,",
          f"{len(inst_entries)} keyed ref entries, {len(inv_entries)} invariant ref entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
