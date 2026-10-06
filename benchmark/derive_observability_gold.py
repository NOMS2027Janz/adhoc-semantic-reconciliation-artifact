#!/usr/bin/env python3
"""Derive and validate the Act-2 gold for the observability case, and check Act-1 consistency.

    python benchmark/derive_observability_gold.py

Act-2 gold is COMPUTED from the oracles in reconcile.observability, so it cannot drift:
  * the verdict gold — act / watch / suppress per (anomaly, context);
  * the incident gold — the correlation of symptoms into incidents, with a probable cause,
    per scenario.

It refuses to write an inconsistent gold. In particular it proves the pragmatic axis is real
(at least one anomaly's verdict CHANGES across contexts — the same data, different meaning), and
that the correlation scenarios have the intended cross-layer structure. It also checks the Act-1
schema gold references only real concept ids and that the ontological false cognate is present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.observability import (verdict_of, correlate, load_act2)   # noqa: E402

CDIR = ROOT / "benchmark" / "cases" / "config_observability"


def main() -> int:
    a2 = load_act2(CDIR)
    anomalies, contexts, scenarios = a2["anomalies"], a2["contexts"], a2["scenarios"]
    errors: list[str] = []

    # --- verdict gold -------------------------------------------------------------------
    verdicts = {}
    for an in anomalies:
        row = {}
        for ctx in contexts:
            row[ctx["id"]] = verdict_of(an, ctx)
        verdicts[an["id"]] = row
    # pragmatic sensitivity: some anomaly's verdict varies with context
    varies = [aid for aid, row in verdicts.items() if len(set(row.values())) > 1]
    if not varies:
        errors.append("no anomaly's verdict changes across contexts; the pragmatic axis is inert")
    # every verdict must be a legal value
    for aid, row in verdicts.items():
        for cid, v in row.items():
            if v not in ("act", "watch", "suppress"):
                errors.append(f"{aid}/{cid}: illegal verdict '{v}'")

    # --- incident (correlation) gold ----------------------------------------------------
    incidents = {}
    for sc in scenarios:
        res = correlate(sc["symptoms"], sc.get("deps", {}), window=sc.get("window", 5.0))
        incidents[sc["id"]] = res["incidents"]
    # sanity on the seeded intent of each scenario
    def n_incidents(sid):
        return len(incidents[sid])
    if "S1" in incidents and n_incidents("S1") != 2:
        errors.append(f"S1 should form 2 incidents (optical+IP together, mgmt separate); got {n_incidents('S1')}")
    if "S2" in incidents and n_incidents("S2") != 2:
        errors.append(f"S2 should form 2 independent incidents; got {n_incidents('S2')}")
    if "S3" in incidents and n_incidents("S3") != 1:
        errors.append(f"S3 should form 1 incident of three; got {n_incidents('S3')}")
    # the cross-layer incident in S1 must be rooted at the optical symptom
    if "S1" in incidents:
        cross = [i for i in incidents["S1"] if len(i["symptoms"]) > 1]
        if not cross or cross[0]["cause"] != "s1":
            errors.append("S1 cross-layer incident should be rooted at the optical symptom s1")

    # --- Act-1 schema gold consistency --------------------------------------------------
    gold = json.loads((CDIR / "gold.json").read_text())
    a_ids = {c["id"] for c in json.loads((CDIR / "model_a_legacy.json").read_text())["concepts"]}
    b_ids = {c["id"] for c in json.loads((CDIR / "model_b_nmop.json").read_text())["concepts"]}
    for c in gold["correspondences"]:
        if c["a"] not in a_ids or c["b"] not in b_ids:
            errors.append(f"correspondence references unknown id: {c}")
    fc_pairs = {(c["a"], c["b"]) for c in gold["false_cognates"]}
    if ("f.alarm", "g.anomaly") not in fc_pairs:
        errors.append("the ontological alarm-vs-anomaly false cognate is missing from the gold")
    # the decomposition: f.alarm must map to more than one NMOP concept
    alarm_targets = [c["b"] for c in gold["correspondences"] if c["a"] == "f.alarm"]
    if len(alarm_targets) < 2:
        errors.append("the legacy alarm should decompose into >1 NMOP concept (alarm-State + fault)")

    if errors:
        print("OBSERVABILITY GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    out = {
        "case": "config_observability",
        "note": "Act-2 gold DERIVED by derive_observability_gold.py from the oracles; do not hand-edit.",
        "verdicts": verdicts,
        "verdict_varies": varies,
        "incidents": incidents,
    }
    (CDIR / "obs_act2_gold.json").write_text(json.dumps(out, indent=2) + "\n")

    print("Wrote config_observability/obs_act2_gold.json")
    print(f"  verdicts: {len(anomalies)} anomalies x {len(contexts)} contexts; "
          f"pragmatic-sensitive anomalies: {varies}")
    for aid, row in verdicts.items():
        print(f"    {aid}: " + "  ".join(f"{c}:{v}" for c, v in row.items()))
    print(f"  correlation: {len(scenarios)} scenarios")
    for sid, incs in incidents.items():
        desc = "; ".join(f"[{'+'.join(i['symptoms'])}->{i['cause']}]" for i in incs)
        print(f"    {sid}: {len(incs)} incident(s)  {desc}")
    print(f"  Act-1: {len(gold['correspondences'])} correspondences (alarm decomposes to "
          f"{alarm_targets}), {len(gold['false_cognates'])} false cognates incl. alarm/anomaly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
