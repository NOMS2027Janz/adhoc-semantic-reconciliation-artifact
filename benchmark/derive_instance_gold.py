#!/usr/bin/env python3
"""Derive and validate instance_hard/instance_gold.json from the hidden `_truth` ids.

    python benchmark/derive_instance_gold.py

Correspondences and native gaps are computed from each individual's `_truth`, so the
gold cannot drift from the case. False cognates, experiment-only flags, oracle answers,
and invariant signatures are read from instance_traps.json and validated for internal
consistency; the script refuses to write an inconsistent gold. In particular it proves
that every experiment-only correspondence is genuinely static-indistinguishable — there
is a same-type, same-attributes, same-topology, keyless twin on both sides — and that the
oracle can separate the twins (distinct interrogation answers), so the case actually
requires live cognition where it claims to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "benchmark" / "cases" / "instance_hard"


def load(name):
    return json.loads((CDIR / name).read_text())


def truth_of(ind_list):
    return {i["id"]: i["_truth"] for i in ind_list}


def static_sig(ind, id2truth):
    """The static signature the agent could key on: type, attrs, neighbour-truths, has-key.
    Name is deliberately excluded (it is the trap surface, not evidence of identity)."""
    neigh = tuple(sorted((r["rel"], id2truth.get(r["target"], r["target"])) for r in ind.get("rels", [])))
    attrs = tuple(sorted((ind.get("attrs") or {}).items()))
    return (ind["type"], attrs, neigh, bool(ind.get("key")))


def main() -> int:
    a = load("individuals_a.json")["individuals"]
    b = load("individuals_b.json")["individuals"]
    traps = load("instance_traps.json")
    errors: list[str] = []

    a_by_id = {i["id"]: i for i in a}
    b_by_id = {i["id"]: i for i in b}
    a_truth, b_truth = truth_of(a), truth_of(b)
    id2truth = {**a_truth, **b_truth}

    # one individual per truth per side
    def one_per_truth(rows, side):
        seen = {}
        for i in rows:
            t = i["_truth"]
            if t in seen:
                errors.append(f"{side}: truth '{t}' on two individuals {seen[t]} and {i['id']}")
            seen[t] = i["id"]
        return seen
    a_of, b_of = one_per_truth(a, "A"), one_per_truth(b, "B")

    # correspondences: truths present on both sides
    shared = sorted(set(a_of) & set(b_of))
    correspondences = [{"a": a_of[t], "b": b_of[t], "truth": t} for t in shared]

    # native gaps: truths on one side only
    a_only = [{"id": a_of[t], "truth": t, "reason": "native gap (A-only)"} for t in sorted(set(a_of) - set(b_of))]
    b_only = [{"id": b_of[t], "truth": t, "reason": "native gap (B-only)"} for t in sorted(set(b_of) - set(a_of))]

    # keys must not collide across different truths (a key is a truthful anchor where present)
    for by_id, truth_map, side in ((a_by_id, a_truth, "A"), (b_by_id, b_truth, "B")):
        keymap = {}
        for i in by_id.values():
            k = i.get("key")
            if not k:
                continue
            if k in keymap and truth_map[keymap[k]] != truth_map[i["id"]]:
                errors.append(f"{side}: key '{k}' shared across truths {keymap[k]}/{i['id']}")
            keymap[k] = i["id"]
    # a key present on both sides must denote the same truth (opaque shared key = identity)
    a_keys = {i["key"]: i["_truth"] for i in a if i.get("key")}
    b_keys = {i["key"]: i["_truth"] for i in b if i.get("key")}
    for k in set(a_keys) & set(b_keys):
        if a_keys[k] != b_keys[k]:
            errors.append(f"shared key '{k}' denotes different truths across sides")

    # false cognates: ids valid, DIFFERENT truth, and NOT a correspondence
    corr_pairs = {(c["a"], c["b"]) for c in correspondences}
    for fc in traps.get("false_cognates", []):
        if fc["a"] not in a_by_id:
            errors.append(f"false cognate a-id not found: {fc['a']}")
        if fc["b"] not in b_by_id:
            errors.append(f"false cognate b-id not found: {fc['b']}")
        if fc["a"] in a_by_id and fc["b"] in b_by_id:
            if a_truth[fc["a"]] == b_truth[fc["b"]]:
                errors.append(f"false cognate {fc['a']}/{fc['b']} share a truth (would be a real merge)")
            if (fc["a"], fc["b"]) in corr_pairs:
                errors.append(f"false cognate {fc['a']}/{fc['b']} is also a correspondence")

    # experiment-only: each is a real correspondence, has a keyless static-twin on both sides,
    # and the oracle separates the twins.
    interr = traps.get("oracle", {}).get("interrogate", {})
    a_sig = {i["id"]: static_sig(i, id2truth) for i in a}
    b_sig = {i["id"]: static_sig(i, id2truth) for i in b}
    for t in traps.get("experiment_only", []):
        if t not in a_of or t not in b_of:
            errors.append(f"experiment-only truth '{t}' is not a both-sides correspondence")
            continue
        ai, bi = a_of[t], b_of[t]
        if a_by_id[ai].get("key") or b_by_id[bi].get("key"):
            errors.append(f"experiment-only '{t}' has a key; then it is statically resolvable")
        # a twin: some OTHER truth whose individuals share the static signature on both sides
        twins = [u for u in shared if u != t
                 and a_sig[a_of[u]] == a_sig[ai] and b_sig[b_of[u]] == b_sig[bi]]
        if not twins:
            errors.append(f"experiment-only '{t}' has no static twin; static evidence would resolve it")
        else:
            for u in twins:
                # the oracle must distinguish t from its twin u on at least one side
                for side_of, byid in ((a_of, a_by_id), (b_of, b_by_id)):
                    it, iu = side_of[t], side_of[u]
                    if interr.get(it) and interr.get(it) == interr.get(iu):
                        errors.append(f"oracle cannot separate experiment-only twins {it}/{iu}")
                if not interr.get(a_of[t]) and not interr.get(b_of[t]):
                    errors.append(f"experiment-only '{t}' has no oracle answer; unresolvable even live")

    # invariant signatures: every service correspondence flagged with invariants is well-formed,
    # and the two name-collision services differ on an invariant (so manipulation can refute).
    inv_by_truth = traps.get("oracle", {}).get("invariants", {})
    for fc in traps.get("false_cognates", []):
        ta, tb = a_truth.get(fc["a"]), b_truth.get(fc["b"])
        if ta in inv_by_truth and tb in inv_by_truth:
            ia, ib = inv_by_truth[ta], inv_by_truth[tb]
            if not any(ia.get(k) != ib.get(k) for k in set(ia) | set(ib)):
                errors.append(f"name-collision {fc['a']}/{fc['b']} share all invariants; "
                              f"a virtual manipulation could not refute the merge")

    if errors:
        print("INSTANCE GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    gold = {
        "case": "instance_hard",
        "operational_case": traps.get("operational_case", ""),
        "seed": traps.get("seed", ""),
        "note": "DERIVED from each individual's hidden _truth by derive_instance_gold.py; do not hand-edit.",
        "correspondences": correspondences,
        "false_cognates": traps.get("false_cognates", []),
        "residual": {"a_only": a_only, "b_only": b_only},
        "experiment_only": traps.get("experiment_only", []),
        "residual_by_placement": traps.get("residual_by_placement", {}),
        "invariants": traps.get("invariants", []),
        "invariant_signatures": inv_by_truth,
        "verification_by_placement": traps.get("verification_by_placement", {}),
    }
    (CDIR / "instance_gold.json").write_text(json.dumps(gold, indent=2) + "\n")
    print("Wrote instance_hard/instance_gold.json")
    print(f"  individuals: A={len(a)} B={len(b)}")
    print(f"  correspondences: {len(correspondences)}  (experiment-only: {len(traps.get('experiment_only', []))})")
    print(f"  false cognates: {len(traps.get('false_cognates', []))}")
    print(f"  native gaps: A-only={len(a_only)} B-only={len(b_only)}")
    print(f"  residual_by_placement: {traps.get('residual_by_placement', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
