#!/usr/bin/env python3
"""Construct the seeded verification set over instance_hard, and derive+validate its gold.

The verification study asks a different question from reconciliation: given a *proposed*
correspondence, does the verifier PASS what is correct and FAIL what is wrong — and by which
mode. The seeded set therefore mixes correct correspondences with wrong ones of two kinds:

  * meaning-visible wrong — the proposal violates an invariant that is visible in the two
    sides' static records (different topology, different service attributes). An invariant
    round-trip can catch these.
  * byte-clean wrong — the two sides have IDENTICAL static records (the crossed mapping of a
    structurally-symmetric or keyless pair), so a byte round-trip passes and an invariant
    round-trip is blind; only a virtual operation (interrogation of an authoritative fact)
    catches them. This is the demonstration the MAGIC paper calls for: a wrong correspondence
    that round-trips.

Everything is built from instance_hard (individuals + oracle), so the invariants, the probe
answers, and the ground truth are all reused. Writes verify_hard/{proposals,verify_gold}.json.
Run derive by executing this file; it validates and refuses an inconsistent set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from reconcile.instance import load_instance_case      # noqa: E402

IN = ROOT / "benchmark" / "cases" / "instance_hard"
OUT = ROOT / "benchmark" / "cases" / "verify_hard"

# proposals: (a_id, b_id, note). truth is derived from instance_hard _truth; category and
# byte-clean-ness are computed. Correct ones should PASS; wrong ones should FAIL.
CORRECT = [
    ("a.r1", "b.roadm1"),      # easy device
    ("a.a1", "b.acc1"),        # easy access
    ("a.cs_odu2", "b.tunnel1"),  # service (provision-confirmable)
    ("a.cs_odu0", "b.tunnel2"),  # service
    ("a.r2", "b.nodeB"),       # symmetric node, CORRECT mapping (serial-confirmable)
    ("a.r3", "b.nodeC"),       # symmetric node, CORRECT mapping
    ("a.omsx", "b.msx"),       # keyless OMS, CORRECT mapping (fibre-confirmable)
    ("a.omsy", "b.msy"),       # keyless OMS, CORRECT mapping
]
WRONG = [
    ("a.svc100", "b.svc100"),  # meaning-visible: same name, different endpoints/capacity
    ("a.r1", "b.r1"),          # meaning-visible: same name, different topology
    ("a.r2", "b.nodeC"),       # BYTE-CLEAN: crossed symmetric mapping (identical records)
    ("a.r3", "b.nodeB"),       # BYTE-CLEAN: crossed symmetric mapping
    ("a.omsx", "b.msy"),       # BYTE-CLEAN: crossed keyless mapping
    ("a.omsy", "b.msx"),       # BYTE-CLEAN: crossed keyless mapping
]


def record(ind):
    """The static record a verifier could read (name is the surface; the rest is evidence)."""
    return {"id": ind["id"], "type": ind["type"], "name": ind["name"],
            "attrs": ind.get("attrs", {}), "topology": ind.get("rels", [])}


def static_key(ind, id2truth):
    """Static signature for byte-clean detection (name excluded — it is the surface)."""
    neigh = tuple(sorted((r["rel"], id2truth.get(r["target"], r["target"]))
                         for r in ind.get("rels", [])))
    attrs = tuple(sorted((ind.get("attrs") or {}).items()))
    return (ind["type"], attrs, neigh)


def main() -> int:
    case = load_instance_case(IN)
    a_by, b_by = case.a_by_id, case.b_by_id
    truth = case.truth
    id2truth = truth
    interr = case.traps.get("oracle", {}).get("interrogate", {})
    inv_by_truth = case.traps.get("oracle", {}).get("invariants", {})
    errors = []

    proposals = []

    def add(a_id, b_id, expect_pass):
        if a_id not in a_by:
            errors.append(f"proposal a-id not found: {a_id}"); return
        if b_id not in b_by:
            errors.append(f"proposal b-id not found: {b_id}"); return
        ia, ib = a_by[a_id], b_by[b_id]
        same_truth = truth[a_id] == truth[b_id]
        if expect_pass and not same_truth:
            errors.append(f"CORRECT proposal {a_id}/{b_id} is not truly the same entity")
        if not expect_pass and same_truth:
            errors.append(f"WRONG proposal {a_id}/{b_id} is actually correct")
        byte_clean = (static_key(ia, id2truth) == static_key(ib, id2truth))
        # which mode can catch a wrong one:
        #  - byte-clean identical records -> only a probe (interrogation/provision) catches
        #  - otherwise the difference is meaning-visible -> an invariant round-trip can catch
        if expect_pass:
            category = "correct"
        elif byte_clean:
            category = "byte-clean"
        else:
            category = "meaning-visible"
        # is a probe available for this pair (does the oracle hold a distinguishing fact)?
        probe = None
        if ia["type"] == "connection-service" and ib["type"] == "connection-service":
            probe = "virtual_provision"
        elif interr.get(a_id) or interr.get(b_id):
            probe = "interrogate"
        proposals.append({
            "id": f"{a_id}__{b_id}", "a_id": a_id, "b_id": b_id,
            "a": record(ia), "b": record(ib),
            "truth_pass": expect_pass, "byte_roundtrips": True,  # all seeded pairs are type-compatible
            "byte_clean_records": byte_clean, "category": category,
            "probe_available": probe,
        })

    for a_id, b_id in CORRECT:
        add(a_id, b_id, True)
    for a_id, b_id in WRONG:
        add(a_id, b_id, False)

    # validate: byte-clean wrong pairs really do have identical records; meaning-visible differ
    for p in proposals:
        if p["category"] == "byte-clean" and not p["byte_clean_records"]:
            errors.append(f"{p['id']} tagged byte-clean but records differ")
        if p["category"] == "meaning-visible" and p["byte_clean_records"]:
            errors.append(f"{p['id']} tagged meaning-visible but records are identical")
        # every wrong pair must be catchable by SOME mode (else the set is unfair)
        if not p["truth_pass"] and p["category"] == "byte-clean" and p["probe_available"] is None:
            errors.append(f"{p['id']} is byte-clean AND has no probe: uncatchable, drop it")

    if errors:
        print("VERIFY SET VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proposals.json").write_text(json.dumps(
        {"case": "verify_hard", "source": "instance_hard",
         "note": "Seeded verification set: correct + meaning-visible-wrong + byte-clean-wrong "
                 "proposals, to measure a verifier's pass-correct/fail-wrong quality by mode.",
         "invariants": case.traps.get("invariants", []),
         "proposals": proposals}, indent=2) + "\n")

    gold = {"case": "verify_hard",
            "note": "DERIVED from instance_hard _truth by build_verify_set.py; do not hand-edit.",
            "verdicts": {p["id"]: {"truth_pass": p["truth_pass"], "category": p["category"],
                                   "byte_clean_records": p["byte_clean_records"],
                                   "probe_available": p["probe_available"]} for p in proposals}}
    (OUT / "verify_gold.json").write_text(json.dumps(gold, indent=2) + "\n")

    n_correct = sum(1 for p in proposals if p["truth_pass"])
    n_mv = sum(1 for p in proposals if p["category"] == "meaning-visible")
    n_bc = sum(1 for p in proposals if p["category"] == "byte-clean")
    print(f"wrote verify_hard: {len(proposals)} proposals "
          f"({n_correct} correct, {n_mv} meaning-visible-wrong, {n_bc} byte-clean-wrong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
