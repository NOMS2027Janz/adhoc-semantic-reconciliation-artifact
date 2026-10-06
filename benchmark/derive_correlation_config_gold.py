#!/usr/bin/env python3
"""Derive and validate the correlation gold for the E7b configuration-domain case.

    python benchmark/derive_correlation_config_gold.py

The gold is COMPUTED from the same correlate() oracle the observability setting uses, so it
cannot drift from the mechanism under test. This script refuses to write a gold whose scenarios
do not have their intended cross-layer structure -- the same guardrail
derive_observability_gold.py applies -- and prints the derived incidents for inspection.

Writes benchmark/cases/correlation_config/corr_gold.json:
  {"incidents": {scenario_id: [{symptoms, cause, layers}]}}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.observability import correlate                    # noqa: E402

CDIR = ROOT / "benchmark" / "cases" / "correlation_config"


def main() -> int:
    scenarios = json.loads((CDIR / "scenarios.json").read_text())["scenarios"]
    incidents = {}
    for sc in scenarios:
        res = correlate(sc["symptoms"], sc.get("deps", {}), window=sc.get("window", 5.0))
        incidents[sc["id"]] = res["incidents"]

    errors: list[str] = []

    def incs(sid):
        return incidents.get(sid, [])

    def parts(sid):
        return {frozenset(i["symptoms"]): i for i in incs(sid)}

    # C1: {s1,s2,s3} rooted at the optical s1, and {s4} separate -> 2 incidents
    if incs("C1"):
        p = parts("C1")
        if len(p) != 2:
            errors.append(f"C1 should form 2 incidents (cross-layer chain + unrelated); got {len(p)}")
        chain = p.get(frozenset({"s1", "s2", "s3"}))
        if not chain:
            errors.append("C1 should merge {s1,s2,s3} into one cross-layer incident")
        elif chain["cause"] != "s1":
            errors.append(f"C1 cross-layer incident should be rooted at optical s1; got {chain['cause']}")
        if frozenset({"s4"}) not in p:
            errors.append("C1 should leave the unrelated service symptom s4 as its own incident")

    # C2: two independent cross-layer incidents {s1,s2} and {s3,s4,s5} -> 2 incidents, no over-merge
    if incs("C2"):
        p = parts("C2")
        if len(p) != 2:
            errors.append(f"C2 should form 2 independent incidents (no over-merge across components); got {len(p)}")
        left = p.get(frozenset({"s1", "s2"}))
        right = p.get(frozenset({"s3", "s4", "s5"}))
        if not left or left["cause"] != "s1":
            errors.append("C2 incident {s1,s2} should exist and be rooted at optical s1")
        if not right or right["cause"] != "s3":
            errors.append("C2 incident {s3,s4,s5} should exist and be rooted at optical s3")

    # C3: the time gate -- {s1,s2} together, s3 (t=30) separate despite same component -> 2 incidents
    if incs("C3"):
        p = parts("C3")
        if len(p) != 2:
            errors.append(f"C3 should form 2 incidents (time gate splits the later service symptom); got {len(p)}")
        if frozenset({"s1", "s2"}) not in p:
            errors.append("C3 should merge the in-window optical/IP pair {s1,s2}")
        if frozenset({"s3"}) not in p:
            errors.append("C3 should keep the out-of-window service symptom s3 separate (time gate)")

    if errors:
        print("CORRELATION_CONFIG GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    out = {
        "case": "correlation_config",
        "note": "Correlation gold DERIVED by derive_correlation_config_gold.py from reconcile.observability.correlate; do not hand-edit.",
        "incidents": incidents,
    }
    (CDIR / "corr_gold.json").write_text(json.dumps(out, indent=2) + "\n")

    print("Wrote correlation_config/corr_gold.json")
    print(f"  correlation: {len(scenarios)} scenarios")
    for sid, incs_ in incidents.items():
        desc = "; ".join(f"[{'+'.join(i['symptoms'])}->{i['cause']}]" for i in incs_)
        print(f"    {sid}: {len(incs_)} incident(s)  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
