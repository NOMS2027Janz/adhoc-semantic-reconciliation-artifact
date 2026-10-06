"""Observability (setting 4): the deterministic pragmatic-verdict and correlation oracles.

Act 1 of this setting reuses the schema-binding harness (two lifted models, a constructed
reference, a gold of correspondences and false cognates) and the instance machinery, so it
needs no new engine. Act 2 is the novel core, and this module carries its two deterministic
oracles, grounded in the NMOP work (RFC 9940 terminology; the anomaly-semantics annotation
set: concern 0-100, confidence 0-100, network plane, pattern, lifecycle, season; the
incident-yang correlation model):

  * ``verdict_of(anomaly, context)`` — the operative verdict for a single anomaly under a
    pragmatic context: **act**, **watch**, or **suppress**. It is a function of the anomaly's
    concern and confidence AND its context (a planned maintenance window makes the same
    deviation benign; an expected seasonal shift on a holiday likewise). The point of the
    setting: the identical anomaly reaches different verdicts under different pragmatics.
  * ``correlate(symptoms, deps, window)`` — genuine multi-symptom cross-layer correlation:
    group symptoms into incidents by resource-dependency and time-proximity, and name each
    incident's probable cause (the root symptom). One optical degradation and the IP loss it
    causes become one incident with one page, not two.

``legacy_verdict`` and ``legacy_correlate`` are the pragmatics-off baselines — the legacy
pipeline that pages every alarm and correlates nothing — for the ON/OFF contrast.
"""
from __future__ import annotations

import json
from pathlib import Path

# verdict thresholds (concern/confidence on the anomaly-semantics 0-100 scales)
ACT_CONCERN, ACT_CONFIDENCE = 70, 60
WATCH_CONCERN = 40
LAYER_ORDER = {"optical": 0, "ip": 1, "service": 1, "control": 2, "management": 3}


# ---------------------------------------------------------------------------------------
# Verdict oracle
# ---------------------------------------------------------------------------------------

def verdict_of(anomaly: dict, context: dict) -> str:
    """Operative verdict for one anomaly under a pragmatic context: act | watch | suppress.

    Context can carry ``maintenance_window`` (a planned change makes the deviation expected)
    and ``season`` (workday | holiday). The anomaly carries ``concern``, ``confidence``,
    ``pattern`` and ``seasonal_expected``."""
    if context.get("maintenance_window"):
        return "suppress"                      # planned change: the deviation is expected
    if (anomaly.get("seasonal_expected") and context.get("season") == "holiday"
            and anomaly.get("pattern") in ("seasonality_shift", "mean_shift")):
        return "suppress"                      # an expected seasonal shift, in its season
    concern = anomaly.get("concern", 0)
    confidence = anomaly.get("confidence", 0)
    if concern >= ACT_CONCERN and confidence >= ACT_CONFIDENCE:
        return "act"
    if concern >= WATCH_CONCERN:
        return "watch"                         # concerning, or concerning-but-unconfident
    return "suppress"


def legacy_verdict(anomaly: dict, context: dict) -> str:
    """The pragmatics-off baseline: the legacy pipeline pages every alarm, blind to context."""
    return "act"


# ---------------------------------------------------------------------------------------
# Correlation oracle (multi-symptom, cross-layer)
# ---------------------------------------------------------------------------------------

def _resource_components(deps: dict) -> dict:
    """Undirected connected components over resources linked by the underlies graph.
    deps maps a resource -> list of resources it underlies (carries)."""
    adj: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for x, unders in deps.items():
        nodes.add(x)
        for y in unders:
            nodes.add(y)
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
    comp: dict[str, int] = {}
    cid = 0
    for n in nodes:
        if n in comp:
            continue
        stack, cid = [n], cid + 1
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp[u] = cid
            stack.extend(adj.get(u, ()))
    return comp


def correlate(symptoms: list[dict], deps: dict, window: float = 5.0) -> dict:
    """Group symptoms into incidents and name each incident's probable cause.

    Two symptoms join the same incident when their resources are in the same
    resource-dependency component AND their times are within ``window``. The probable cause
    is the incident's root symptom: lowest layer (optical before IP before control/management),
    earliest in time. Returns {"incidents": [{"symptoms": [ids], "cause": id, "layers": [...]}]}.
    """
    comp = _resource_components(deps)
    n = len(symptoms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = symptoms[i]["resource"], symptoms[j]["resource"]
            same_component = comp.get(ri) is not None and comp.get(ri) == comp.get(rj)
            close_in_time = abs(symptoms[i].get("t", 0) - symptoms[j].get("t", 0)) <= window
            if same_component and close_in_time:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    incidents = []
    for members in groups.values():
        def rootkey(i):
            return (LAYER_ORDER.get(symptoms[i].get("layer", "ip"), 9), symptoms[i].get("t", 0))
        root = min(members, key=rootkey)
        incidents.append({
            "symptoms": sorted(symptoms[i]["id"] for i in members),
            "cause": symptoms[root]["id"],
            "layers": sorted({symptoms[i].get("layer", "ip") for i in members}),
        })
    incidents.sort(key=lambda inc: inc["symptoms"])
    return {"incidents": incidents}


def legacy_correlate(symptoms: list[dict], deps: dict, window: float = 5.0) -> dict:
    """The pragmatics-off baseline: no correlation — every symptom is its own alarm/incident."""
    return {"incidents": [{"symptoms": [s["id"]], "cause": s["id"], "layers": [s.get("layer", "ip")]}
                          for s in symptoms]}


# ---------------------------------------------------------------------------------------
# Case loading (Act 2 payloads)
# ---------------------------------------------------------------------------------------

def load_act2(case_dir: str | Path) -> dict:
    p = Path(case_dir)

    def j(name):
        f = p / name
        return json.loads(f.read_text()) if f.exists() else {}

    return {
        "anomalies": j("anomalies.json").get("anomalies", []),
        "contexts": j("contexts.json").get("contexts", []),
        "scenarios": j("scenarios.json").get("scenarios", []),
    }
