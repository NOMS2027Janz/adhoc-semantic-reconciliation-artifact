#!/usr/bin/env python3
"""Derive and validate a case's gold.json from the two models' reference bindings.

    python benchmark/derive_gold.py <case_name>

Correspondences and residual are computed from each concept's `ref` (so they
cannot drift from the models); false cognates, invariants, and verification are
read from the case's traps.json. The script validates consistency and refuses to
write a gold that is internally inconsistent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import SemanticModel  # noqa: E402


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else "config_big_hard"
    cdir = ROOT / "benchmark" / "cases" / case
    a = SemanticModel.from_json(sorted(cdir.glob("model_a*.json"))[0])
    b = SemanticModel.from_json(sorted(cdir.glob("model_b*.json"))[0])
    traps = json.loads((cdir / "traps.json").read_text())

    errors: list[str] = []

    # one concept per non-null ref per side
    def ref_index(m):
        idx: dict[str, str] = {}
        for c in m.concepts:
            if c.ref:
                if c.ref in idx:
                    errors.append(f"{m.system}: ref '{c.ref}' used by {idx[c.ref]} and {c.id}")
                idx[c.ref] = c.id
        return idx

    ai, bi = ref_index(a), ref_index(b)
    shared = sorted(set(ai) & set(bi))
    correspondences = [{"a": ai[r], "b": bi[r], "ref": r} for r in shared]

    a_ids = {c.id for c in a.concepts}
    b_ids = {c.id for c in b.concepts}
    matched_a = {c["a"] for c in correspondences}
    matched_b = {c["b"] for c in correspondences}
    opaque = set(traps.get("opaque", []))

    def residual(m, matched, opaque_ids):
        out = []
        for c in m.concepts:
            if c.id in matched:
                continue
            reason = "opaque (no public meaning)" if c.ref is None else "native gap (self-extend)"
            out.append({"id": c.id, "ref": c.ref, "reason": reason})
        return out

    res_a = residual(a, matched_a, opaque)
    res_b = residual(b, matched_b, opaque)

    # validate false cognates: valid ids, and the two sides bind to DIFFERENT refs
    for fc in traps.get("false_cognates", []):
        if fc["a"] not in a_ids:
            errors.append(f"false cognate a-id not found: {fc['a']}")
        if fc["b"] not in b_ids:
            errors.append(f"false cognate b-id not found: {fc['b']}")
        ca = a.by_id.get(fc["a"]); cb = b.by_id.get(fc["b"])
        if ca and cb and ca.ref is not None and cb.ref is not None and ca.ref == cb.ref:
            errors.append(f"false cognate {fc['a']}/{fc['b']} share ref '{ca.ref}' (would be a real match)")
        # a false cognate must NOT be a correct correspondence
        if {"a": fc["a"], "b": fc["b"]} in [{"a": c["a"], "b": c["b"]} for c in correspondences]:
            errors.append(f"false cognate {fc['a']}/{fc['b']} is also a correspondence")

    # every opaque id should be present and unmatched (residual)
    for oid in opaque:
        if oid not in a_ids and oid not in b_ids:
            errors.append(f"opaque id not found: {oid}")
        if oid in matched_a or oid in matched_b:
            errors.append(f"opaque id {oid} is matched but should be residual")

    if errors:
        print("GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    gold = {
        "case": case,
        "operational_case": traps.get("operational_case", "configuration"),
        "seed": traps.get("seed", ""),
        "note": "Gold standard DERIVED from model ref bindings by derive_gold.py; do not hand-edit.",
        "correspondences": correspondences,
        "false_cognates": traps.get("false_cognates", []),
        "residual": {"a_only": res_a, "b_only": res_b},
        "residual_by_placement": traps.get("residual_by_placement", {}),
        "invariants": traps.get("invariants", []),
        "verification_by_placement": traps.get("verification_by_placement", {}),
    }
    (cdir / "gold.json").write_text(json.dumps(gold, indent=2) + "\n")
    print(f"Wrote {case}/gold.json")
    print(f"  concepts: A={len(a.concepts)} B={len(b.concepts)}")
    print(f"  correspondences: {len(correspondences)}")
    print(f"  false cognates: {len(traps.get('false_cognates', []))}")
    print(f"  residual: A={len(res_a)} (opaque {sum(1 for r in res_a if r['ref'] is None)}), "
          f"B={len(res_b)} (opaque {sum(1 for r in res_b if r['ref'] is None)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
