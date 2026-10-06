#!/usr/bin/env python3
"""Breadth cases for the observability setting: the 'an alarm is not an anomaly' reconciliation
in two further fault domains - IP/routing (obs_routing) and compute/server (obs_compute).

The NMOP ladder (RFC 9940: anomaly / symptom / fault / alarm / problem / cause / incident /
concern / confidence) is the fixed standard, so what varies across these cases is the fault
domain: the legacy side and the concrete alarms/anomalies. Each case keeps the setting's core
mechanism - the ontological cognate (a declared undesirable STATE vs a DEVIATION that may be
benign), the one-to-many decomposition of the overloaded legacy alarm, and the two pins
(fixed severity != dynamic concern score; static probable-cause != correlation-derived cause).

Writes model_a_legacy.json, model_b_nmop.json, gold.json (hand-authored structure) and
reference.json per case. Validate with:  python benchmark/pack.py validate obs_routing obs_compute
and check the schema controls with:  python pipeline/run.py --case obs_routing --no-write
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# domain-specific concrete content; the concept STRUCTURE is shared across domains
DOMAINS = {
    "obs_routing": dict(
        legacy_dialect="RFC 8632 / syslog / SNMP / X.733 (routing)",
        nmop_note="IP/routing fault domain",
        node_label="managed-object", node_syn=["resource", "router", "BGP session"],
        node_ex="router R7; the BGP session to AS64500", node_inst=["R7", "bgp-R7-AS64500"],
        event_ex="a BGP state-change trap on the session at t0", event_inst=["trap-bgp-R7-t0"],
        alarm_ex="a MAJOR 'session-down' alarm on the BGP session to AS64500",
        alarm_inst=["alarm-bgp-R7-down"],
        cause_inst=["bgpConnectionLost", "holdTimerExpired"],
        anomaly_ex="a rising route-churn rate on R7 that may be a benign reconvergence",
        anomaly_inst=["churn-R7-rising"],
        symptom_ex="repeated best-path changes for a prefix set (a manifestation)",
        fault_ex="the session to AS64500 is down (a confirmed fault)",
        problem_ex="the peering to AS64500 is unstable (a diagnosed problem)",
        cause_ex="a flapping upstream link, derived by correlating the churn and the resets",
        incident_ex="the AS64500 peering incident grouping the session and churn anomalies",
        concern_ex="a concern score of 0.8 for the churn during business hours",
        confidence_ex="0.6 confidence that the churn is service-affecting"),
    "obs_compute": dict(
        legacy_dialect="RFC 8632 / syslog / SNMP / X.733 (compute)",
        nmop_note="compute/server fault domain",
        node_label="managed-object", node_syn=["resource", "server", "container"],
        node_ex="server node-14; the app container on it", node_inst=["node-14", "ctr-app-node14"],
        event_ex="a threshold-crossing trap for CPU on node-14 at t0", event_inst=["trap-cpu-node14-t0"],
        alarm_ex="a CRITICAL 'over-temperature' alarm on node-14",
        alarm_inst=["alarm-node14-overtemp"],
        cause_inst=["thermalTrip", "fanFailure"],
        anomaly_ex="a rising CPU-saturation trend on node-14 that may be an expected batch job",
        anomaly_inst=["cpu-node14-rising"],
        symptom_ex="elevated request latency on the app container (a manifestation)",
        fault_ex="a failed cooling fan on node-14 (a confirmed fault)",
        problem_ex="node-14 is thermally throttling under load (a diagnosed problem)",
        cause_ex="a failed fan, derived by correlating the temperature and fan-speed readings",
        incident_ex="the node-14 thermal incident grouping the temperature and CPU anomalies",
        concern_ex="a concern score of 0.9 for the saturation outside the batch window",
        confidence_ex="0.7 confidence that the saturation is user-affecting"),
}


def legacy_model(d):
    return {
        "system": "Agent F - legacy fault manager",
        "dialect": d["legacy_dialect"],
        "modules": ["x733-probable-cause", "rfc8632-alarms"],
        "note": "Lifted legacy model. An alarm is a catch-all, bundling an event, an undesirable "
                "state, a fixed severity and a static probable-cause, all hard-coded at emission. "
                "ref=null marks a native concept with no shared category.",
        "concepts": [
            {"id": "f.node", "label": d["node_label"], "synonyms": d["node_syn"], "kind": "resource",
             "gloss": "a managed element the fault manager watches", "example": d["node_ex"],
             "ref": "node", "relations": [], "instances": d["node_inst"]},
            {"id": "f.event", "label": "event", "synonyms": ["syslog", "trap"], "kind": "event",
             "gloss": "a syslog message or SNMP trap: something happened on a resource at a moment",
             "example": d["event_ex"], "ref": "event",
             "relations": [{"rel": "on", "target": "f.node"}], "instances": d["event_inst"]},
            {"id": "f.alarm", "label": "alarm", "synonyms": ["fault alarm", "alarm condition"],
             "kind": "alarm",
             "gloss": "an emitted alarm: an undesirable condition (a state) on a resource that also "
                      "implies a fault occurrence, carrying a fixed severity and a probable cause; the "
                      "catch-all that bundles what NMOP separates",
             "example": d["alarm_ex"], "ref": "alarm-state",
             "relations": [{"rel": "on", "target": "f.node"}, {"rel": "carries", "target": "f.severity"},
                           {"rel": "carries", "target": "f.probable_cause"}], "instances": d["alarm_inst"]},
            {"id": "f.severity", "label": "severity", "synonyms": ["perceived severity", "alarm level"],
             "kind": "severity",
             "gloss": "the fixed severity label assigned at emission: critical, major, minor or warning "
                      "- static, not recomputed from context",
             "example": "MAJOR", "ref": None, "relations": [{"rel": "of", "target": "f.alarm"}],
             "instances": ["critical", "major", "minor", "warning"]},
            {"id": "f.probable_cause", "label": "probable-cause", "synonyms": ["X.733 cause", "cause code"],
             "kind": "cause-code",
             "gloss": "a static probable-cause code from the X.733 dictionary, fixed at emission - not "
                      "derived by correlating events",
             "example": "X.733 '" + d["cause_inst"][0] + "'", "ref": None,
             "relations": [{"rel": "of", "target": "f.alarm"}], "instances": d["cause_inst"]},
            {"id": "f.clear", "label": "clear", "synonyms": ["alarm clear"], "kind": "clear",
             "gloss": "a notification that a previously raised alarm condition has cleared",
             "example": "a clear for " + d["alarm_inst"][0], "ref": None,
             "relations": [{"rel": "clears", "target": "f.alarm"}], "instances": ["clear-" + d["alarm_inst"][0]]},
        ],
    }


def nmop_model(d):
    C = lambda i, l, s, k, g, ex, ref, rel, inst: {
        "id": i, "label": l, "synonyms": s, "kind": k, "gloss": g, "example": ex, "ref": ref,
        "relations": rel, "instances": inst}
    return {
        "system": "Agent G - NMOP agent",
        "dialect": "IETF NMOP (RFC 9940 + anomaly-semantics / incident-yang)",
        "modules": ["rfc9940-anomaly", "nmop-anomaly-semantics", "nmop-incident"],
        "note": "Lifted NMOP model, " + d["nmop_note"] + ". The RFC 9940 ladder separates what the "
                "legacy alarm conflates: an anomaly is a DEVIATION that may be benign; a fault is a "
                "confirmed occurrence; an alarm is a declared undesirable state; each anomaly carries a "
                "dynamic concern and confidence. ref=null marks a native concept with no legacy category.",
        "concepts": [
            C("g.node", "resource", ["managed object", "network element"], "resource",
              "a managed element under observation", d["node_ex"], "node", [], d["node_inst"]),
            C("g.event", "event", ["notification"], "event",
              "a raw event on a resource at a moment, before interpretation", d["event_ex"], "event",
              [{"rel": "on", "target": "g.node"}], d["event_inst"]),
            C("g.anomaly", "anomaly", ["deviation"], "anomaly",
              "a deviation from expected behaviour that MAY be benign - not itself an undesirable state",
              d["anomaly_ex"], "anomaly", [{"rel": "on", "target": "g.node"}], d["anomaly_inst"]),
            C("g.symptom", "symptom", ["manifestation"], "symptom",
              "an observable manifestation of an underlying condition; a native NMOP notion", d["symptom_ex"],
              None, [{"rel": "of", "target": "g.anomaly"}], ["symptom-1"]),
            C("g.fault", "fault", ["failure occurrence"], "fault",
              "a confirmed fault occurrence - the thing that has actually gone wrong", d["fault_ex"], "fault",
              [{"rel": "on", "target": "g.node"}], ["fault-1"]),
            C("g.alarm", "alarm", ["alarm state"], "alarm",
              "a declared undesirable STATE requiring attention, distinct from the raw deviation", d["alarm_ex"],
              "alarm-state", [{"rel": "on", "target": "g.node"}], ["alarm-1"]),
            C("g.problem", "problem", ["diagnosis"], "problem",
              "a diagnosed problem behind one or more faults; a native NMOP notion", d["problem_ex"], None,
              [{"rel": "explains", "target": "g.fault"}], ["problem-1"]),
            C("g.cause", "cause", ["root cause"], "cause",
              "a cause derived by CORRELATING events and anomalies, not a static code", d["cause_ex"], "cause",
              [{"rel": "of", "target": "g.problem"}], ["cause-1"]),
            C("g.incident", "incident", ["grouping"], "incident",
              "an incident grouping related anomalies, faults and alarms; a native NMOP notion", d["incident_ex"],
              None, [{"rel": "groups", "target": "g.anomaly"}], ["incident-1"]),
            C("g.concern", "concern-score", ["concern"], "concern",
              "a dynamic concern score annotating an anomaly, recomputed from context - not a fixed severity",
              d["concern_ex"], "concern", [{"rel": "annotates", "target": "g.anomaly"}], ["0.8", "0.9"]),
            C("g.confidence", "confidence-score", ["confidence"], "confidence",
              "a confidence score on an anomaly's assessment; a native NMOP annotation", d["confidence_ex"],
              None, [{"rel": "annotates", "target": "g.anomaly"}], ["0.6", "0.7"]),
        ],
    }


def gold_for(case):
    return {
        "case": case,
        "operational_case": "observability (legacy fault vs IETF NMOP, no common model)",
        "seed": "hand-authored breadth case, grounded in RFC 9940 and the NMOP anomaly-semantics / incident drafts",
        "note": "Schema gold. Correspondences include the one-to-many DECOMPOSITION of the overloaded "
                "legacy alarm into its NMOP constituents (alarm-state and fault). The headline false "
                "cognate is ONTOLOGICAL: a legacy alarm is not an NMOP anomaly - a declared undesirable "
                "State vs a deviation that may be benign. Two pins: fixed severity is not a dynamic "
                "concern score; a static probable-cause is not a correlation-derived cause.",
        "correspondences": [
            {"a": "f.node", "b": "g.node", "ref": "node"},
            {"a": "f.event", "b": "g.event", "ref": "event"},
            {"a": "f.alarm", "b": "g.alarm", "ref": "alarm-state"},
            {"a": "f.alarm", "b": "g.fault", "ref": "fault"},
        ],
        "false_cognates": [
            {"a": "f.alarm", "b": "g.anomaly",
             "why": "the ONTOLOGICAL cognate: a legacy alarm is a declared undesirable STATE; an NMOP "
                    "anomaly is a DEVIATION that may be benign. Same intuition, different kind of thing."},
            {"a": "f.severity", "b": "g.concern",
             "why": "a fixed severity assigned at emission is not the dynamic, context-recomputed concern score."},
            {"a": "f.probable_cause", "b": "g.cause",
             "why": "a static X.733 probable-cause code is not a cause derived by correlating events."},
        ],
        "residual": {
            "a_only": [{"id": "f.clear", "ref": None, "closure": "legacy-native alarm-clear"}],
            "b_only": [{"id": "g.symptom", "ref": None}, {"id": "g.problem", "ref": None},
                       {"id": "g.incident", "ref": None}, {"id": "g.confidence", "ref": None}],
        },
        "residual_by_placement": {"both_cognitive": 1, "one_inert": 3, "both_inert": 5},
        "invariants": ["ontology (state vs deviation)", "severity-vs-concern",
                       "probable-cause-vs-cause", "alarm-decomposition"],
        "verification_by_placement": {
            "both_cognitive": "worked anomaly run and read-back: correlate a live anomaly and confirm the decomposition and pins hold",
            "one_inert": "worked read-back on the live side; reference pins on the inert side",
            "both_inert": "external adjudication"},
    }


def reference_for(case):
    return {
        "id": f"ref.{case}.adhoc.v1", "kind": "lexical",
        "note": "Thin ad-hoc reference the agents construct. Distinct entries for alarm-state, anomaly, "
                "fault, concern and cause, so the ontological cognate (alarm vs anomaly) and the two pins "
                "bind apart. The NMOP-native concepts (symptom, problem, incident, confidence) have NO "
                "entry and remain residual.",
        "entries": [
            {"id": "node", "label": "resource", "synonyms": ["managed object", "network element"], "class": "resource",
             "definition": "A managed element under observation.", "example": "a router, session, server, or container"},
            {"id": "event", "label": "event", "synonyms": ["notification", "trap", "syslog"], "class": "event",
             "definition": "A raw event on a resource at a moment, before interpretation.", "example": "a threshold-crossing trap"},
            {"id": "alarm-state", "label": "alarm", "synonyms": ["alarm state", "alarm condition"], "class": "state",
             "definition": "A declared undesirable STATE requiring attention.", "example": "a raised alarm on a resource"},
            {"id": "fault", "label": "fault", "synonyms": ["failure occurrence"], "class": "occurrence",
             "definition": "A confirmed fault occurrence - the thing that has actually gone wrong.", "example": "a session down; a failed fan"},
            {"id": "anomaly", "label": "anomaly", "synonyms": ["deviation"], "class": "deviation",
             "definition": "A deviation from expected behaviour that MAY be benign; not itself an undesirable state.", "example": "a rising trend that may be expected load"},
            {"id": "concern", "label": "concern score", "synonyms": ["concern"], "class": "annotation",
             "definition": "A dynamic concern score annotating an anomaly, recomputed from context; not a fixed severity.", "example": "0.8 during business hours"},
            {"id": "cause", "label": "cause", "synonyms": ["root cause"], "class": "derived",
             "definition": "A cause derived by correlating events and anomalies; not a static code.", "example": "a flapping upstream link, correlated"},
        ],
    }


def main() -> int:
    for case, d in DOMAINS.items():
        cdir = ROOT / "benchmark" / "cases" / case
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "model_a_legacy.json").write_text(json.dumps(legacy_model(d), indent=2) + "\n")
        (cdir / "model_b_nmop.json").write_text(json.dumps(nmop_model(d), indent=2) + "\n")
        (cdir / "gold.json").write_text(json.dumps(gold_for(case), indent=2) + "\n")
        (cdir / "reference.json").write_text(json.dumps(reference_for(case), indent=2) + "\n")
        print(f"wrote {case}: 6 legacy + 11 NMOP concepts, 4 correspondences (1->2 decomposition), 3 cognates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
