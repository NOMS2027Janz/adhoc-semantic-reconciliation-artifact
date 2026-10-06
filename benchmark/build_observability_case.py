#!/usr/bin/env python3
"""Construct the seeded observability case (setting 4), grounded in the NMOP work.

Emits, under benchmark/cases/:
  config_observability/model_a_legacy.json, model_b_nmop.json, reference.json, gold.json
      — Act 1: the schema binding (legacy fault manager vs IETF NMOP), reference anchored to
        RFC 9940, the ontological alarm-vs-anomaly false cognate, the one-to-many decomposition
        of the overloaded legacy alarm, the severity-vs-concern and probable-cause-vs-cause pins.
  config_observability/anomalies.json, contexts.json, scenarios.json
      — Act 2: annotated anomalies (concern/confidence/plane/pattern/lifecycle per the
        anomaly-semantics draft), the pragmatic contexts (the matrix), and the multi-symptom
        cross-layer correlation scenarios. The verdict and incident gold are DERIVED by
        derive_observability_gold.py from the oracles in reconcile.observability.
  obs_instance/ — a small instance sub-case: which legacy alarm and which NMOP anomaly are the
        same underlying condition (resource + time), reused by the instance agent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "benchmark" / "cases" / "config_observability"
IDIR = ROOT / "benchmark" / "cases" / "obs_instance"

# ---- Act 1: the two lifted models -----------------------------------------------------------
LEGACY = {
    "system": "Agent F — legacy fault manager",
    "dialect": "RFC 8632 / syslog / SNMP / X.733",
    "modules": ["x733-probable-cause", "rfc8632-alarms"],
    "note": "Lifted legacy model. An alarm is a catch-all, bundling an event, an undesirable "
            "state, a fixed severity and a static probable-cause, all hard-coded at emission. "
            "Fields: label, synonyms, kind, gloss, example, the ad-hoc reference entry bound "
            "(ref; null = native/no shared category), relations, instances.",
    "concepts": [
        {"id": "f.node", "label": "managed-object", "synonyms": ["resource", "network element"],
         "kind": "resource", "gloss": "a managed network element the fault manager watches",
         "example": "router R1; wavelength lambda1", "ref": "node",
         "relations": [], "instances": ["R1", "lambda1"]},
        {"id": "f.event", "label": "event", "synonyms": ["syslog", "trap"], "kind": "event",
         "gloss": "a syslog message or SNMP trap: something happened on a resource at a moment",
         "example": "a threshold-crossing trap on lambda1 at t0", "ref": "event",
         "relations": [{"rel": "on", "target": "f.node"}], "instances": ["trap-lambda1-t0"]},
        {"id": "f.alarm", "label": "alarm", "synonyms": ["fault alarm", "alarm condition"],
         "kind": "alarm", "gloss": "an emitted alarm: an undesirable condition (a state) on a "
         "resource that also implies a fault occurrence, carrying a fixed severity and a probable "
         "cause; the catch-all that bundles what NMOP separates",
         "example": "a MAJOR 'signal-degrade' alarm on lambda1", "ref": "alarm-state",
         "relations": [{"rel": "on", "target": "f.node"}, {"rel": "carries", "target": "f.severity"},
                       {"rel": "carries", "target": "f.probable_cause"}],
         "instances": ["alarm-lambda1-SD"]},
        {"id": "f.severity", "label": "severity", "synonyms": ["perceived severity", "alarm level"],
         "kind": "severity", "gloss": "the fixed severity label assigned at emission: critical, "
         "major, minor or warning — static, not recomputed from context",
         "example": "MAJOR", "ref": None,
         "relations": [{"rel": "of", "target": "f.alarm"}], "instances": ["critical", "major", "minor", "warning"]},
        {"id": "f.probable_cause", "label": "probable-cause", "synonyms": ["X.733 cause", "cause code"],
         "kind": "cause-code", "gloss": "a static probable-cause code from the X.733 dictionary, "
         "fixed at emission — not derived by correlating events",
         "example": "X.733 'lossOfSignal'", "ref": None,
         "relations": [{"rel": "of", "target": "f.alarm"}], "instances": ["lossOfSignal", "degradedSignal"]},
        {"id": "f.clear", "label": "clear", "synonyms": ["alarm clear"], "kind": "clear",
         "gloss": "a notification that a previously raised alarm condition has cleared",
         "example": "clear for alarm-lambda1-SD", "ref": None,
         "relations": [{"rel": "clears", "target": "f.alarm"}], "instances": ["clear-lambda1-SD"]},
    ],
}

NMOP = {
    "system": "Agent G — IETF NMOP agent",
    "dialect": "RFC 9940 + anomaly-semantics + incident-yang",
    "modules": ["rfc9940-terminology", "nmop-anomaly-semantics", "nmop-network-incident-yang"],
    "note": "Lifted NMOP model. RFC 9940 separates what legacy conflates: event, anomaly, "
            "symptom, fault, alarm (a State), problem (a State), cause, incident; plus the "
            "anomaly-semantics annotations (concern, confidence). Many concepts have no legacy "
            "equivalent (the legacy alarm swallows them). Same field conventions as the legacy model.",
    "concepts": [
        {"id": "g.node", "label": "resource", "synonyms": ["managed resource", "node"], "kind": "resource",
         "gloss": "a resource whose characteristics are observed", "example": "wavelength lambda1",
         "ref": "node", "relations": [], "instances": ["R1", "lambda1"]},
        {"id": "g.event", "label": "event", "synonyms": ["value variation"], "kind": "event",
         "gloss": "the variation in the value of a characteristic of a resource at a distinct "
         "moment in time (RFC 9940)", "example": "a pre-FEC BER sample at t0", "ref": "event",
         "relations": [{"rel": "on", "target": "g.node"}], "instances": ["ber-sample-lambda1-t0"]},
        {"id": "g.anomaly", "label": "anomaly", "synonyms": ["deviation", "outlier"], "kind": "anomaly",
         "gloss": "an unusual or unexpected event or pattern that DEVIATES from normal, expected "
         "behaviour (RFC 9940) — a deviation, NOT by definition undesirable; it may be perfectly benign",
         "example": "a rising pre-FEC BER trend on lambda1 that may be benign during maintenance",
         "ref": "anomaly", "relations": [{"rel": "on", "target": "g.node"}], "instances": ["anom-ber-lambda1"]},
        {"id": "g.symptom", "label": "symptom", "synonyms": ["indication"], "kind": "symptom",
         "gloss": "an observable value/state/event considered an indication of a problem or "
         "potential problem (RFC 9940)", "example": "the BER anomaly taken as a symptom of degradation",
         "ref": None, "relations": [{"rel": "indicates", "target": "g.problem"}], "instances": ["sym-ber-lambda1"]},
        {"id": "g.fault", "label": "fault", "synonyms": ["defect"], "kind": "fault",
         "gloss": "an occurrence (event or change) that is not desired/required, as it may indicate "
         "a current or future undesired state (RFC 9940)", "example": "the fibre degradation itself",
         "ref": "fault", "relations": [{"rel": "on", "target": "g.node"}], "instances": ["fault-lambda1-degrade"]},
        {"id": "g.alarm", "label": "alarm", "synonyms": ["alarm state"], "kind": "alarm",
         "gloss": "an undesirable STATE in a resource that requires corrective action — a state in "
         "its own right (RFC 9940)", "example": "the signal-degrade alarm state on lambda1",
         "ref": "alarm-state", "relations": [{"rel": "on", "target": "g.node"}], "instances": ["alarm-state-lambda1"]},
        {"id": "g.problem", "label": "problem", "synonyms": ["undesirable state"], "kind": "problem",
         "gloss": "a state that is undesirable and that may require remedial action (RFC 9940)",
         "example": "the degraded-service problem state", "ref": None,
         "relations": [], "instances": ["problem-svc-degrade"]},
        {"id": "g.cause", "label": "cause", "synonyms": ["root cause"], "kind": "cause",
         "gloss": "the events, established by CORRELATION, that gave rise to a fault or problem "
         "(RFC 9940) — derived, not a static code", "example": "the optical BER symptom found to be "
         "the root of the IP loss", "ref": "cause", "relations": [{"rel": "of", "target": "g.problem"}],
         "instances": ["cause-optical-ber"]},
        {"id": "g.incident", "label": "incident", "synonyms": ["correlated incident"], "kind": "incident",
         "gloss": "an undesired occurrence such as interruption, degradation or below-target "
         "performance of a service, formed by correlating symptoms (RFC 9940 / incident-yang)",
         "example": "one incident correlating the optical and IP symptoms", "ref": None,
         "relations": [{"rel": "correlates", "target": "g.symptom"}], "instances": ["incident-0042"]},
        {"id": "g.concern", "label": "concern-score", "synonyms": ["concern", "significance"],
         "kind": "concern", "gloss": "a DYNAMIC concern score (0-100) reflecting the current degree "
         "of concern given context (anomaly-semantics) — recomputed, not a fixed label",
         "example": "concern 78 for the BER trend now", "ref": "concern",
         "relations": [{"rel": "annotates", "target": "g.anomaly"}], "instances": ["concern-78"]},
        {"id": "g.confidence", "label": "confidence-score", "synonyms": ["confidence"], "kind": "confidence",
         "gloss": "a confidence score (0-100): the detector's confidence in the anomaly classification "
         "(anomaly-semantics)", "example": "confidence 82", "ref": None,
         "relations": [{"rel": "annotates", "target": "g.anomaly"}], "instances": ["confidence-82"]},
    ],
}

REFERENCE = {
    "id": "ref.observability.adhoc.v1", "kind": "lexical", "reference_type": "ad_hoc_by_example",
    "note": "An ad-hoc reference the two agents construct, ANCHORED TO THE RFC 9940 term ladder — "
            "the standard that separates what legacy conflates. One entry per shared category the "
            "exchange needs, each fixed by a canonical example. It deliberately does not cover the "
            "NMOP concepts legacy has no equivalent for (anomaly-vs-alarm aside, which it pins "
            "precisely because that is the dangerous cognate).",
    "entries": [
        {"id": "node", "label": "resource", "synonyms": ["managed object", "network element"],
         "class": "resource", "definition": "A managed network element whose characteristics are observed.",
         "example": "wavelength lambda1"},
        {"id": "event", "label": "event", "synonyms": ["value variation"], "class": "event",
         "definition": "The variation in the value of a characteristic of a resource at a distinct "
         "moment (RFC 9940 Event).", "example": "a pre-FEC BER sample at t0"},
        {"id": "alarm-state", "label": "alarm", "synonyms": ["alarm state"], "class": "state",
         "definition": "An undesirable STATE in a resource requiring corrective action — a state in "
         "its own right (RFC 9940 Alarm).", "example": "the signal-degrade alarm state on lambda1"},
        {"id": "fault", "label": "fault", "synonyms": ["defect"], "class": "occurrence",
         "definition": "An occurrence (event or change) that is not desired/required (RFC 9940 Fault).",
         "example": "the fibre degradation underlying the alarm"},
        {"id": "anomaly", "label": "anomaly", "synonyms": ["deviation"], "class": "deviation",
         "definition": "An unusual or unexpected event or pattern that DEVIATES from normal expected "
         "behaviour (RFC 9940 Anomaly) — a deviation, NOT by definition undesirable, and therefore "
         "NOT the same as an alarm (which is a declared undesirable state).",
         "example": "a rising BER trend that may be benign in a maintenance window"},
        {"id": "concern", "label": "concern score", "synonyms": ["significance"], "class": "annotation",
         "definition": "A dynamic concern score (0-100) computed from current context "
         "(anomaly-semantics) — NOT a fixed severity label assigned at emission.",
         "example": "concern 78 now, lower during a maintenance window"},
        {"id": "cause", "label": "cause", "synonyms": ["root cause"], "class": "derived",
         "definition": "The events, established by CORRELATION, that gave rise to a fault or problem "
         "(RFC 9940 Cause) — NOT a static probable-cause code fixed at emission.",
         "example": "the optical BER symptom found to be the root of the IP loss"},
    ],
}

GOLD = {
    "case": "config_observability",
    "operational_case": "observability (legacy fault vs IETF NMOP, no common model)",
    "seed": "hand-authored, grounded in RFC 9940 and the NMOP anomaly-semantics / incident-yang drafts",
    "note": "Act-1 schema gold. Correspondences include the one-to-many DECOMPOSITION of the "
            "overloaded legacy alarm into its NMOP constituents (alarm-State and fault). The "
            "headline false cognate is ONTOLOGICAL: a legacy alarm is not an NMOP anomaly — an "
            "alarm is a declared undesirable State, an anomaly is a deviation that may be benign. "
            "Two further pins: a fixed severity is not a dynamic concern score; a static "
            "probable-cause is not a correlation-derived cause.",
    "correspondences": [
        {"a": "f.node", "b": "g.node", "ref": "node"},
        {"a": "f.event", "b": "g.event", "ref": "event"},
        {"a": "f.alarm", "b": "g.alarm", "ref": "alarm-state"},
        {"a": "f.alarm", "b": "g.fault", "ref": "fault"}
    ],
    "false_cognates": [
        {"a": "f.alarm", "b": "g.anomaly",
         "why": "the ONTOLOGICAL cognate: a legacy alarm is a declared undesirable STATE; an NMOP "
                "anomaly is a DEVIATION that may be benign. Same intuition ('something abnormal'), "
                "different kind of thing. Must not be corresponded."},
        {"a": "f.severity", "b": "g.concern",
         "why": "a fixed severity label assigned at emission is not the dynamic, context-recomputed "
                "concern score of anomaly-semantics."},
        {"a": "f.probable_cause", "b": "g.cause",
         "why": "a static X.733 probable-cause code is not a cause derived by correlating events."}
    ],
    "residual": {
        "a_only": [{"id": "f.clear", "ref": None, "closure": "legacy-native alarm-clear"}],
        "b_only": [{"id": "g.symptom", "ref": None}, {"id": "g.problem", "ref": None},
                   {"id": "g.incident", "ref": None}, {"id": "g.confidence", "ref": None}]
    },
    "residual_by_placement": {"both_cognitive": 1, "one_inert": 3, "both_inert": 5},
    "invariants": ["ontology (state vs deviation)", "severity-vs-concern", "probable-cause-vs-cause",
                   "alarm-decomposition"],
    "verification_by_placement": {
        "both_cognitive": "worked anomaly run and read-back: correlate a live anomaly and confirm the "
                          "decomposition and pins hold",
        "one_inert": "worked read-back on the live side; reference pins on the inert side",
        "both_inert": "external adjudication"
    },
}

# ---- Act 2: annotated anomalies (verdict phase) ---------------------------------------------
# anomaly-semantics annotations: concern/confidence (0-100), plane, pattern, lifecycle, seasonal.
ANOMALIES = [
    {"id": "A1", "label": "rising pre-FEC BER on lambda1", "resource": "lambda1", "plane": "forwarding",
     "pattern": "trend", "lifecycle": "validation", "concern": 78, "confidence": 82,
     "seasonal_expected": False, "action_reason_trigger": ["degrade", "corrupt", "link-layer"]},
    {"id": "A2", "label": "minor latency jitter on L1", "resource": "L1", "plane": "forwarding",
     "pattern": "spike", "lifecycle": "detection", "concern": 35, "confidence": 60,
     "seasonal_expected": False, "action_reason_trigger": ["delay", "congestion", "queue"]},
    {"id": "A3", "label": "BGP session flaps on R1", "resource": "R1", "plane": "control",
     "pattern": "other", "lifecycle": "validation", "concern": 88, "confidence": 90,
     "seasonal_expected": False, "action_reason_trigger": ["drop", "unreachable", "next-hop"]},
    {"id": "A4", "label": "CPU utilisation drift on R9 mgmt", "resource": "R9", "plane": "management",
     "pattern": "mean_shift", "lifecycle": "detection", "concern": 30, "confidence": 45,
     "seasonal_expected": False, "action_reason_trigger": ["none", "none", "none"]},
    {"id": "A5", "label": "traffic seasonality shift on L2", "resource": "L2", "plane": "forwarding",
     "pattern": "seasonality_shift", "lifecycle": "refinement", "concern": 55, "confidence": 62,
     "seasonal_expected": True, "action_reason_trigger": ["none", "administered", "schedule"]},
    {"id": "A6", "label": "unconfirmed spike on lambda2", "resource": "lambda2", "plane": "forwarding",
     "pattern": "spike", "lifecycle": "detection", "concern": 80, "confidence": 30,
     "seasonal_expected": False, "action_reason_trigger": ["degrade", "unknown", "unknown"]},
]

CONTEXTS = [
    {"id": "normal", "maintenance_window": False, "season": "workday",
     "note": "business as usual"},
    {"id": "maintenance", "maintenance_window": True, "season": "workday",
     "note": "a planned maintenance change is in effect on the resource"},
    {"id": "holiday", "maintenance_window": False, "season": "holiday",
     "note": "a holiday: seasonal shifts are expected"},
]

# ---- Act 2: multi-symptom cross-layer correlation scenarios ----------------------------------
# deps: resource -> resources it underlies (carries). Symptoms carry resource/layer/time.
SCENARIOS = [
    {"id": "S1",
     "note": "an optical degradation and the IP loss it causes should be ONE incident, with an "
             "unrelated management symptom left separate",
     "deps": {"lambda1": ["L1"]},
     "symptoms": [
         {"id": "s1", "label": "pre-FEC BER rising", "resource": "lambda1", "layer": "optical", "t": 0},
         {"id": "s2", "label": "IP packet loss", "resource": "L1", "layer": "ip", "t": 2},
         {"id": "s3", "label": "mgmt CPU drift", "resource": "R9", "layer": "management", "t": 1},
     ]},
    {"id": "S2",
     "note": "two IP symptoms on independent links sharing no underlay should be TWO incidents",
     "deps": {"lambdaA": ["LA"], "lambdaB": ["LB"]},
     "symptoms": [
         {"id": "s1", "label": "loss on LA", "resource": "LA", "layer": "ip", "t": 0},
         {"id": "s2", "label": "loss on LB", "resource": "LB", "layer": "ip", "t": 1},
     ]},
    {"id": "S3",
     "note": "an optical degradation under two IP links it carries should be ONE incident of three, "
             "rooted at the optical symptom",
     "deps": {"lambda3": ["L3a", "L3b"]},
     "symptoms": [
         {"id": "s1", "label": "pre-FEC BER rising", "resource": "lambda3", "layer": "optical", "t": 0},
         {"id": "s2", "label": "loss on L3a", "resource": "L3a", "layer": "ip", "t": 2},
         {"id": "s3", "label": "loss on L3b", "resource": "L3b", "layer": "ip", "t": 3},
     ]},
]


# ---- instance sub-case: which legacy alarm and which NMOP anomaly are the same condition ------
def build_instance_subcase():
    ACOND = "underlying-condition"
    ents = [
        dict(truth="c1", type=ACOND, key="corr-001",
             a=dict(id="f.al1", name="alarm-lambda1-SD", attrs={"resource": "lambda1", "t": "t0"}),
             b=dict(id="g.an1", name="anom-ber-lambda1", attrs={"resource": "lambda1", "t": "t0"})),
        dict(truth="c2", type=ACOND, key="corr-002",
             a=dict(id="f.al2", name="alarm-R1-bgp", attrs={"resource": "R1", "t": "t1"}),
             b=dict(id="g.an2", name="anom-bgp-R1", attrs={"resource": "R1", "t": "t1"})),
        # experiment-only twins: same resource+time on two sub-interfaces; separable only by an
        # interrogated correlation-id (a live probe), not by the static resource+time.
        dict(truth="c3", type=ACOND, experiment_only=True,
             a=dict(id="f.al3", name="alarm-L3-loss", attrs={"resource": "L3", "t": "t2"}),
             b=dict(id="g.an3", name="anom-L3-a", attrs={"resource": "L3", "t": "t2"})),
        dict(truth="c4", type=ACOND, experiment_only=True,
             a=dict(id="f.al4", name="alarm-L3-loss2", attrs={"resource": "L3", "t": "t2"}),
             b=dict(id="g.an4", name="anom-L3-b", attrs={"resource": "L3", "t": "t2"})),
    ]

    def side(side):
        rows = []
        for e in ents:
            s = e.get(side)
            if not s:
                continue
            rows.append({"id": s["id"], "type": e["type"], "name": s["name"], "key": e.get("key"),
                         "attrs": s.get("attrs", {}), "rels": [], "_truth": e["truth"]})
        return rows

    a_rows, b_rows = side("a"), side("b")
    a_of = {e["truth"]: e["a"]["id"] for e in ents}
    b_of = {e["truth"]: e["b"]["id"] for e in ents}
    shared = sorted(set(a_of) & set(b_of))
    corr = [{"a": a_of[t], "b": b_of[t], "truth": t} for t in shared]
    eo = [e["truth"] for e in ents if e.get("experiment_only")]
    interr = {"f.al3": {"correlation_id": "CID-3A"}, "g.an3": {"correlation_id": "CID-3A"},
              "f.al4": {"correlation_id": "CID-3B"}, "g.an4": {"correlation_id": "CID-3B"}}
    IDIR.mkdir(parents=True, exist_ok=True)
    (IDIR / "individuals_a.json").write_text(json.dumps(
        {"system": "Agent F (legacy alarms)", "dialect": "RFC 8632", "individuals": a_rows}, indent=2) + "\n")
    (IDIR / "individuals_b.json").write_text(json.dumps(
        {"system": "Agent G (NMOP anomalies)", "dialect": "RFC 9940", "individuals": b_rows}, indent=2) + "\n")
    traps = {"case": "obs_instance", "operational_case": "observability (alarm/anomaly co-reference)",
             "seed": "hand-authored alarm-vs-anomaly co-reference sub-case", "false_cognates": [],
             "experiment_only": eo, "oracle": {"interrogate": interr, "invariants": {}},
             "residual_by_placement": {"both_cognitive": 0, "one_inert": 2, "both_inert": 2},
             "invariants": [], "verification_by_placement": {
                 "both_cognitive": "interrogate the correlation-id", "one_inert": "live side only",
                 "both_inert": "static evidence only"}}
    (IDIR / "instance_traps.json").write_text(json.dumps(traps, indent=2) + "\n")
    gold = {"case": "obs_instance", "note": "DERIVED inline from _truth.", "correspondences": corr,
            "false_cognates": [], "residual": {"a_only": [], "b_only": []}, "experiment_only": eo,
            "residual_by_placement": traps["residual_by_placement"], "invariants": [],
            "invariant_signatures": {}, "verification_by_placement": traps["verification_by_placement"]}
    (IDIR / "instance_gold.json").write_text(json.dumps(gold, indent=2) + "\n")
    (IDIR / "instance_reference.json").write_text(json.dumps(
        {"note": "opaque correlation keys binding an alarm and an anomaly to one condition.",
         "instance": [{"key": e["key"], "type": e["type"], "canonical": e["a"]["name"]}
                      for e in ents if e.get("key")], "invariant": []}, indent=2) + "\n")
    return len(a_rows), len(b_rows), len(corr), len(eo)


def main() -> int:
    CDIR.mkdir(parents=True, exist_ok=True)
    (CDIR / "model_a_legacy.json").write_text(json.dumps(LEGACY, indent=2) + "\n")
    (CDIR / "model_b_nmop.json").write_text(json.dumps(NMOP, indent=2) + "\n")
    (CDIR / "reference.json").write_text(json.dumps(REFERENCE, indent=2) + "\n")
    (CDIR / "gold.json").write_text(json.dumps(GOLD, indent=2) + "\n")
    (CDIR / "anomalies.json").write_text(json.dumps({"anomalies": ANOMALIES}, indent=2) + "\n")
    (CDIR / "contexts.json").write_text(json.dumps({"contexts": CONTEXTS}, indent=2) + "\n")
    (CDIR / "scenarios.json").write_text(json.dumps({"scenarios": SCENARIOS}, indent=2) + "\n")
    ia, ib, ic, ieo = build_instance_subcase()
    print("wrote config_observability:",
          f"{len(LEGACY['concepts'])} legacy + {len(NMOP['concepts'])} NMOP concepts,",
          f"{len(GOLD['correspondences'])} correspondences, {len(GOLD['false_cognates'])} false cognates;",
          f"Act2: {len(ANOMALIES)} anomalies x {len(CONTEXTS)} contexts, {len(SCENARIOS)} correlation scenarios")
    print("wrote obs_instance:", f"A={ia} B={ib}, {ic} correspondences, {ieo} experiment-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
