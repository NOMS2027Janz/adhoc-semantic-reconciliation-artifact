"""Observability Act-2 stacks: the pragmatic verdict and the multi-symptom correlation.

Act 1 reuses the schema and instance stacks unchanged. Act 2 is the novel core and needs two
small structured-output stacks, both using the proven ``chat.completions.parse`` mechanism:

  * ``VerdictStack`` — given a set of anomalies (with their anomaly-semantics annotations) under
    a pragmatic context, decide the operative verdict for each: **act**, **watch**, or
    **suppress**. With pragmatics OFF it sees only a legacy alarm view (a severity, no concern/
    confidence, no context) — the legacy pipeline, which pages blind to context.
  * ``CorrelationStack`` — given a set of symptoms across layers with a resource-dependency map,
    group them into incidents and name each incident's probable cause. With pragmatics OFF it
    sees only bare alarms with no dependencies — and cannot correlate.

Both are injectable-client for offline testing. The point of the setting is the ON/OFF contrast:
whether cognition that carries the pragmatics reaches the verdict and the correlation the legacy
pipeline cannot.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import BaseModel


class _Verdict(BaseModel):
    anomaly_id: str
    verdict: str            # act | watch | suppress


class VerdictResult(BaseModel):
    verdicts: list[_Verdict]


class _Incident(BaseModel):
    symptoms: list[str]
    cause: str


class CorrelationResult(BaseModel):
    incidents: list[_Incident]


VERDICT_SYSTEM_ON = (
    "You triage network anomalies for an operations centre. Each anomaly carries semantic "
    "annotations (per the IETF NMOP anomaly-semantics work): a concern score and a confidence "
    "score (each 0-100), a network plane, a detected pattern, a lifecycle stage, and whether it is "
    "an expected seasonal effect. You are also given the current CONTEXT — notably whether a "
    "planned maintenance window is in effect on the resource, and whether it is a workday or a "
    "holiday. For each anomaly decide the operative verdict:\n"
    " - \"act\": raise a page now — the anomaly is concerning and the detector is confident;\n"
    " - \"watch\": hold and observe — concerning but not confident, or moderate;\n"
    " - \"suppress\": do not page — the deviation is benign or EXPECTED in this context (a planned "
    "maintenance window makes a deviation expected; an expected seasonal shift in its season is "
    "expected).\n"
    "Judge each anomaly on its concern, its confidence, AND its context — the SAME deviation can "
    "warrant a page in normal operation and none during planned maintenance. Return a verdict for "
    "every anomaly."
)

VERDICT_SYSTEM_OFF = (
    "You are a legacy fault console. You receive alarms, each with a fixed severity "
    "(critical/major/minor/warning). You have no concern or confidence scores and no context. "
    "For each alarm decide \"act\" (page), \"watch\", or \"suppress\" from what you have. Return a "
    "verdict for every alarm."
)

CORR_SYSTEM_ON = (
    "You correlate network symptoms into incidents. You are given symptoms across layers (optical, "
    "IP, control, management), each on a resource at a time, and a RESOURCE-DEPENDENCY map saying "
    "which resource underlies (carries) which. Group into incidents: two symptoms belong to the "
    "same incident when their resources are in the same dependency chain and their times are close, "
    "because one condition is causing the other across layers. For each incident, name the probable "
    "CAUSE — the root symptom, at the lowest layer (optical before IP before control/management) and "
    "earliest in time. An optical degradation and the IP loss it causes are ONE incident with ONE "
    "cause, not two pages. Return the incidents, each as its list of symptom ids and its cause id."
)

CORR_SYSTEM_OFF = (
    "You are a legacy fault console. You receive alarms, each on a resource. You have no "
    "dependency information and no correlation. Report each alarm as its own incident (its own "
    "page), with itself as the cause. Return one incident per alarm."
)


def _severity(concern: int) -> str:
    return ("critical" if concern >= 70 else "major" if concern >= 50
            else "minor" if concern >= 35 else "warning")


def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class ObsReconciliation:
    phase: str
    pragmatics: bool
    context_id: str = ""
    scenario_id: str = ""
    verdicts: dict = field(default_factory=dict)          # anomaly_id -> verdict
    incidents: list = field(default_factory=list)          # [{symptoms:[..], cause:..}]
    effort: dict = field(default_factory=dict)
    submitted: bool = False


class _BaseObsStack:
    def __init__(self, model: str, pragmatics: bool = True, client=None):
        self.model = model
        self.pragmatics = pragmatics
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _call(self, system, user, schema):
        client = self._get_client()
        t0 = time.time()
        completion = client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(user, indent=2)}],
            response_format=schema)
        elapsed = time.time() - t0
        result = completion.choices[0].message.parsed
        usage = getattr(completion, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        effort = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None),
                  "total_tokens": getattr(usage, "total_tokens", None),
                  "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                  "latency_s": round(elapsed, 2), "model": self.model}
        return result, effort


class VerdictStack(_BaseObsStack):
    def reconcile(self, anomalies: list[dict], context: dict) -> ObsReconciliation:
        rec = ObsReconciliation(phase="verdict", pragmatics=self.pragmatics, context_id=context.get("id", ""))
        if self.pragmatics:
            system = VERDICT_SYSTEM_ON
            user = {"context": {"maintenance_window": context.get("maintenance_window", False),
                                "season": context.get("season", "workday")},
                    "anomalies": [{"anomaly_id": a["id"], "label": a["label"], "resource": a["resource"],
                                   "plane": a["plane"], "pattern": a["pattern"], "lifecycle": a["lifecycle"],
                                   "concern": a["concern"], "confidence": a["confidence"],
                                   "seasonal_expected": a.get("seasonal_expected", False)}
                                  for a in anomalies]}
        else:
            system = VERDICT_SYSTEM_OFF
            user = {"alarms": [{"anomaly_id": a["id"], "label": a["label"], "resource": a["resource"],
                                "severity": _severity(a["concern"])} for a in anomalies]}
        result, rec.effort = self._call(system, user, VerdictResult)
        valid = {a["id"] for a in anomalies}
        for v in (result.verdicts if result else []):
            if v.anomaly_id in valid:
                val = v.verdict.strip().lower()
                if val in ("act", "watch", "suppress"):
                    rec.verdicts[v.anomaly_id] = val
        rec.submitted = bool(rec.verdicts)
        return rec


class CorrelationStack(_BaseObsStack):
    def reconcile(self, scenario: dict) -> ObsReconciliation:
        rec = ObsReconciliation(phase="correlation", pragmatics=self.pragmatics,
                                scenario_id=scenario.get("id", ""))
        syms = scenario["symptoms"]
        if self.pragmatics:
            system = CORR_SYSTEM_ON
            user = {"symptoms": [{"id": s["id"], "label": s["label"], "resource": s["resource"],
                                  "layer": s["layer"], "time": s.get("t", 0)} for s in syms],
                    "resource_dependencies": scenario.get("deps", {})}
        else:
            system = CORR_SYSTEM_OFF
            user = {"alarms": [{"id": s["id"], "label": s["label"], "resource": s["resource"]} for s in syms]}
        result, rec.effort = self._call(system, user, CorrelationResult)
        valid = {s["id"] for s in syms}
        for inc in (result.incidents if result else []):
            members = [m for m in inc.symptoms if m in valid]
            if members:
                cause = inc.cause if inc.cause in valid else members[0]
                rec.incidents.append({"symptoms": sorted(members), "cause": cause})
        rec.incidents.sort(key=lambda i: i["symptoms"])
        rec.submitted = bool(rec.incidents)
        return rec
