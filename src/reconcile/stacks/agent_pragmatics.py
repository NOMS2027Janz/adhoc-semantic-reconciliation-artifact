"""Pragmatics stack: attributing REALM and AUTHORITY across the cross-domain seam.

The first three components of a reconciliation — lexical, structural, instance — settle
*what* two concepts mean and *whether* they correspond. They do not settle *whose realm
owns* a shared field: when a Cascade (IP) service rides a Meridian (transport) circuit, the
committed rate, the realised latency, the path protection, and the demarcation are each seen
from both domains, and provisioning across the seam is only correct if the AUTHORITATIVE
source of truth for each is respected — the realm that owns and realises the field's value,
whose value must govern when the two sides disagree. This is the pragmatic component the
first two settings held fixed; here it is measured.

The stack presents both bespoke models (masked by the cognition placement) and, optionally,
the ad-hoc reference, and asks the model to attribute each seam field to X (transport), Y
(IP), or shared. It reuses the proven ``chat.completions.parse`` structured-output mechanism
of the schema stack. The client is injectable, so the loop can be driven by a scripted fake
with no API.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from reconcile.model import Case
from reconcile.reference import Reference


class _Attribution(BaseModel):
    field_id: str
    authority: str          # "X" | "Y" | "shared"
    rationale: str


class PragmaticsResult(BaseModel):
    attributions: list[_Attribution]


SYSTEM = (
    "Two bespoke systems from DIFFERENT domains meet at one seam: a Cascade (IP/VPN) service "
    "rides a Meridian (transport) circuit as its underlay. Agent X is Meridian (transport — "
    "circuits, bearers, physical paths). Agent Y is Cascade (IP/VPN — services, attachments, "
    "traffic classes).\n\n"
    "Corresponding the concepts is not enough. For each shared requirement FIELD at the seam, "
    "decide the AUTHORITATIVE SOURCE OF TRUTH: which realm OWNS the field's true value — the "
    "realm that realises it, and whose value must govern if the two sides disagree.\n"
    " - answer \"X\" if the transport realm (Meridian) is the source of truth;\n"
    " - answer \"Y\" if the IP realm (Cascade) is the source of truth;\n"
    " - answer \"shared\" if it is a single co-owned point that neither side's local naming "
    "overrides.\n\n"
    "Judge by which realm the value physically belongs to, NOT by which side happens to mention "
    "it: a requirement one side merely STATES is not owned by that side if the other realm "
    "REALISES it, and the realm that carries a value does not thereby own a value another realm "
    "commits. This is a pragmatic judgement of realm and authority, separate from matching the "
    "concepts. Return an authority for every field."
)

PLACEMENT_NOTE = {
    "both_cognitive": "\n\nBoth systems are live and can explain their own view; weigh both "
                      "realms' claims and decide who actually owns each field.",
    "one_inert": "\n\nOne system is inert and cannot explain itself — infer its realm and its "
                 "ownership from its structure, its instances, and the shared reference if given.",
    "both_inert": "\n\nNeither system can explain itself; decide each field's realm from the "
                  "structure, the instances, and the shared reference alone.",
}

REFERENCE_NOTE = (
    "\n\nYou are given the ad-hoc reference the two agents constructed. Its definitions pin what "
    "each field means and often imply which realm realises it (e.g. a value defined as a property "
    "of the transport path, or as the customer's service commitment). Use it to decide authority."
)


def _concept_view(c, inert: bool) -> dict:
    out = {"id": c.id, "label": c.label, "kind": c.kind}
    if c.relations:
        out["relations"] = [{"rel": r.get("rel"), "target": r.get("target")} for r in c.relations]
    if c.instances:
        out["instances"] = list(c.instances)
    if not inert:                       # a live side volunteers its self-explanation
        if c.gloss:
            out["gloss"] = c.gloss
        if c.example:
            out["example"] = c.example
    return out


def _ref_entry(e) -> dict:
    return {"id": e.id, "label": e.label, "definition": getattr(e, "definition", ""),
            "example": getattr(e, "example", "")}


@dataclass
class PragmaticsReconciliation:
    placement: str
    uses_reference: bool
    attributions: dict = field(default_factory=dict)     # field_id -> authority
    rationales: dict = field(default_factory=dict)
    effort: dict = field(default_factory=dict)
    submitted: bool = False


class PragmaticsStack:
    def __init__(self, case: Case, fields: list[dict], use_reference: bool = False,
                 model: str | None = None, inert_side: str = "b", client=None):
        self.case = case
        self.fields = fields
        self.uses_reference = use_reference
        self.model = model or "gpt-5.6"
        self.inert_side = inert_side          # 'b' = Cascade (Y) inert in one_inert
        self.name = f"pragmatics({'ref' if use_reference else 'no-ref'})"
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _inert_sides(self, placement):
        if placement == "both_cognitive":
            return False, False
        if placement == "both_inert":
            return True, True
        return (self.inert_side == "a"), (self.inert_side == "b")   # one_inert

    def _payload(self, placement) -> str:
        a_inert, b_inert = self._inert_sides(placement)
        payload = {
            "agent_X_meridian": {"domain": "transport", "inert": a_inert,
                                 "concepts": [_concept_view(c, a_inert) for c in self.case.model_a.concepts]},
            "agent_Y_cascade": {"domain": "IP/VPN", "inert": b_inert,
                                "concepts": [_concept_view(c, b_inert) for c in self.case.model_b.concepts]},
            "seam_fields": [{"field_id": f["id"], "pins": f.get("pins", {}),
                             "reference_category": f.get("ref")} for f in self.fields],
        }
        if self.uses_reference:
            payload["ad_hoc_reference"] = [_ref_entry(e) for e in self.case.reference.entries]
        return json.dumps(payload, indent=2)

    def reconcile(self, placement: str = "both_cognitive") -> PragmaticsReconciliation:
        client = self._get_client()
        system = SYSTEM + PLACEMENT_NOTE.get(placement, "")
        if self.uses_reference:
            system += REFERENCE_NOTE
        rec = PragmaticsReconciliation(placement=placement, uses_reference=self.uses_reference)
        t0 = time.time()
        completion = client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": self._payload(placement)}],
            response_format=PragmaticsResult,
        )
        elapsed = time.time() - t0
        result = completion.choices[0].message.parsed
        valid = {f["id"] for f in self.fields}
        for a in (result.attributions if result else []):
            if a.field_id in valid:
                auth = a.authority.strip().lower()
                auth = {"x": "X", "y": "Y", "shared": "shared", "both": "shared"}.get(auth, a.authority)
                rec.attributions[a.field_id] = auth
                rec.rationales[a.field_id] = a.rationale
        rec.submitted = bool(rec.attributions)
        usage = getattr(completion, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        rec.effort = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "latency_s": round(elapsed, 2), "model": self.model,
        }
        return rec
