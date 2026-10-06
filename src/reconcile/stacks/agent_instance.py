"""Instance-reconciliation agent: a bounded tool-use loop over the live-cognition oracle.

Unlike the schema stack (a single prompt in, an alignment out), instance co-reference
needs the agent to *act on the graph*: to interrogate a live individual for an
authoritative fact, and to exercise a proposed correspondence by a virtual
provision-and-read-back. So this stack runs a loop in which the oracle (see
``reconcile.instance.Oracle``) is exposed to the model as tools, and the model calls
them until it submits an alignment or the turn budget is spent.

It uses the OpenAI **Responses API** (``client.responses.create``): the gpt-5 reasoning
models permit function tools there but not on chat/completions (which rejects tools
alongside reasoning). Reasoning items returned by the model are fed back into the next
turn so the model can continue a chain of tool calls.

The oracle is deterministic and reads the hidden truth; the model sees only what a call
returns. Availability of the tools follows the cognition placement (the oracle enforces
it), and every call is counted. The whole exchange — every tool call, every answer, the
final rationale — is captured as a transcript, so a run can be *shown*, not only scored.

The client is injectable, so the loop can be driven by a scripted fake with no API.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from reconcile.instance import InstanceCase, Oracle

# evidence factors that may be masked from an individual's record (id is always present)
EVIDENCE_ALL = ("name", "key", "attrs", "rels")


@dataclass
class InstanceReconciliation:
    placement: str
    reference_variant: str
    proposed: list[frozenset] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)          # frozenset pair -> confidence
    residual_a: list[str] = field(default_factory=list)
    residual_b: list[str] = field(default_factory=list)
    confirmed_pairs: set = field(default_factory=set)        # provision-confirmed (mechanical)
    oracle_calls: dict = field(default_factory=dict)         # {interrogate, provision}
    interrogated: set = field(default_factory=set)
    effort: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    submitted: bool = False


def _view(ind: dict, evidence: set) -> dict:
    out = {"id": ind["id"]}                                  # record identity: always present
    if "name" in evidence:
        out["name"] = ind["name"]
    if "key" in evidence and ind.get("key"):
        out["key"] = ind["key"]
    if "attrs" in evidence and ind.get("attrs"):
        out["attrs"] = ind["attrs"]
    if "rels" in evidence and ind.get("rels"):
        out["topology"] = ind["rels"]
    return out


SYSTEM_PREAMBLE = (
    "You reconcile INDIVIDUALS (specific entities) across two populated graphs of ONE "
    "network. The schema is already reconciled, so types are comparable; your job is "
    "entity resolution: decide which individual in graph A is the SAME real-world entity "
    "as which individual in graph B. Each individual has a record id (always present) and "
    "some of: a local name, an opaque shared key, attributes, and topology (edges to other "
    "individuals). Two traps recur: distinct individuals can share a local name (do NOT "
    "merge them), and two individuals that look identical in their static record may be "
    "different entities. Rules: never merge two distinct individuals; hold every decision "
    "with a confidence in [0,1]; and where identity cannot be settled from the evidence, "
    "leave the individual in the residual (refer it onward) rather than guess."
)

# Per-tool descriptions, assembled into the prompt according to the interrogation level so the
# prompt advertises exactly the tools the agent is actually given (the "reach" it has).
TOOL_DESC = {
    "interrogate_discovery":
        " - interrogate(individual_id, attribute=optional): ask for one authoritative fact "
        "about a live individual, or omit the attribute to receive all available facts. If you "
        "name an attribute the individual does not have, the reply lists the attributes it does "
        "have. Use it to separate individuals whose static records are identical.",
    "interrogate_lookup":
        " - interrogate(individual_id, attribute): ask for one authoritative fact about a live "
        "individual by naming the attribute you want (e.g. serial). You must name the attribute; "
        "there is no way to list an individual's available facts. Use it to separate individuals "
        "whose static records are identical.",
    "provision":
        " - virtual_provision(a_id, b_id): exercise a candidate correspondence by provisioning "
        "a service through it and reading the objects back, confirming or refuting it against "
        "the semantic invariants (endpoint identity, connectivity, capacity, layering, "
        "switching, multiplexing). Only services can be provisioned. Use it to CONFIRM a "
        "service correspondence and to catch same-name services that are actually different.",
    "submit":
        " - submit_alignment(correspondences, residual_a, residual_b): submit your final "
        "answer. correspondences is a list of {a_id, b_id, confidence}.",
}

CLOSER_TOOLS = (
    "\n\nWork by proposing, then verifying with the tools where you can, then submitting. "
    "Prefer a confirmed correspondence to a guessed one; an unconfirmed guess belongs in "
    "the residual. Call submit_alignment exactly once, when you are done."
)
CLOSER_NOTOOLS = (
    "\n\nYou have no tools to probe the network: decide identity from the static records above "
    "alone. Where a record is underdetermined, leave the individual in the residual rather than "
    "guess. Call submit_alignment exactly once, when you are done."
)

PLACEMENT_NOTE = {
    "both_cognitive": "\n\nBOTH graphs are live: any tools you are given act on either side.",
    "one_inert": "\n\nGraph B is INERT — a static snapshot. Any tools you are given act ONLY "
                 "on graph A (the live side); for B rely on its static record and any published "
                 "reference. Individuals on B that are separable only by a live probe cannot be "
                 "confirmed — refer them onward.",
    "both_inert": "\n\nBOTH graphs are INERT — static snapshots. The oracle is UNAVAILABLE; "
                  "no interrogation or provision is possible. Propose from static evidence "
                  "only, keep confidences modest, and leave underdetermined individuals in the "
                  "residual for external adjudication.",
}

REFERENCE_NOTE = {
    "none": "",
    "instance": "\n\nYou are given an INSTANCE REFERENCE: a published table binding opaque "
                "keys to canonical descriptors. Two individuals that carry the SAME key are "
                "the same entity; use it to settle keyed individuals directly.",
    "invariant": "\n\nYou are given an INVARIANT REFERENCE: published semantic-invariant "
                 "signatures per opaque key. Use it to check a candidate service correspondence "
                 "against the published invariants WITHOUT a live provision — useful when a side "
                 "is inert and cannot be exercised.",
}


def _fn_tool(name, description, properties, required):
    # Responses API function-tool shape (flat; no nested 'function' object).
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


_INTERROGATE_DISCOVERY = _fn_tool(
    "interrogate",
    "Ask one authoritative fact about a live individual: name an attribute (e.g. "
    "serial), or omit it to receive all available facts. Naming an absent attribute "
    "returns the list of interrogable attributes.",
    {"individual_id": {"type": "string"}, "attribute": {"type": "string"}},
    ["individual_id"])
_INTERROGATE_LOOKUP = _fn_tool(
    "interrogate",
    "Ask one authoritative fact about a live individual by naming the attribute you want "
    "(e.g. serial). You must name the attribute; the individual's available facts are not "
    "listed for you.",
    {"individual_id": {"type": "string"}, "attribute": {"type": "string"}},
    ["individual_id", "attribute"])
_PROVISION = _fn_tool(
    "virtual_provision",
    "Provision a service through a candidate correspondence and read back the "
    "invariant check (confirm/refute). One id from each side.",
    {"a_id": {"type": "string"}, "b_id": {"type": "string"}},
    ["a_id", "b_id"])
_SUBMIT = _fn_tool(
    "submit_alignment",
    "Submit the final co-reference alignment.",
    {"correspondences": {"type": "array", "items": {
        "type": "object", "properties": {
            "a_id": {"type": "string"}, "b_id": {"type": "string"},
            "confidence": {"type": "number"}},
        "required": ["a_id", "b_id", "confidence"], "additionalProperties": False}},
     "residual_a": {"type": "array", "items": {"type": "string"}},
     "residual_b": {"type": "array", "items": {"type": "string"}}},
    ["correspondences", "residual_a", "residual_b"])

TOOLS = [_INTERROGATE_DISCOVERY, _PROVISION, _SUBMIT]   # legacy full repertoire
SUBMIT_ONLY = [_SUBMIT]

# The interrogation "reach" axis: what the agent may DO on a live side, from no probing to
# the full act-on-the-network repertoire. Each level names the tools it exposes and the
# prompt snippets that advertise them. "full" reproduces the study's default behaviour.
INTERROGATION_LEVELS = ("none", "lookup", "discovery", "full")


def _tools_for(interrogation: str):
    """Live-side tool list for an interrogation level (submit_alignment always included)."""
    if interrogation == "none":
        return [_SUBMIT]
    if interrogation == "lookup":
        return [_INTERROGATE_LOOKUP, _SUBMIT]
    if interrogation == "discovery":
        return [_INTERROGATE_DISCOVERY, _SUBMIT]
    return [_INTERROGATE_DISCOVERY, _PROVISION, _SUBMIT]   # "full"


def _tools_note(interrogation: str) -> str:
    """The prompt block advertising exactly the tools the interrogation exposes."""
    if interrogation == "none":
        return CLOSER_NOTOOLS
    parts = ["\n\nYou have tools that act on the network where a side is live:"]
    if interrogation == "lookup":
        parts.append(TOOL_DESC["interrogate_lookup"])
    else:
        parts.append(TOOL_DESC["interrogate_discovery"])
    if interrogation == "full":
        parts.append(TOOL_DESC["provision"])
    parts.append(TOOL_DESC["submit"])
    return "\n".join(parts) + CLOSER_TOOLS


def _g(obj, key, default=None):
    """Read a field from an SDK object or a plain dict (the fake client)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class InstanceAgentStack:
    def __init__(self, case: InstanceCase, model: str, reference_variant: str = "none",
                 evidence: set | None = None, inert_side: str = "b",
                 budget: int | None = 30, max_turns: int | None = None,
                 interrogation: str = "full", client=None):
        self.case = case
        self.model = model
        self.reference_variant = reference_variant
        self.evidence = set(EVIDENCE_ALL if evidence is None else evidence)
        self.inert_side = inert_side
        self.budget = budget
        if interrogation not in INTERROGATION_LEVELS:
            raise ValueError(f"interrogation must be one of {INTERROGATION_LEVELS}, got {interrogation!r}")
        self.interrogation = interrogation
        # generous turn budget: a probe-happy weak model needs room to interrogate every
        # ambiguous individual and still submit. Too tight a cap makes it run out of turns
        # mid-probe and never submit. Min 12; 24 when the oracle budget is unbounded.
        self.max_turns = max_turns or max(12, (budget if budget is not None else 18) + 6)
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _user_payload(self) -> str:
        payload = {
            "graph_A": {"system": "Agent T",
                        "individuals": [_view(i, self.evidence) for i in self.case.a]},
            "graph_B": {"system": "Agent I",
                        "individuals": [_view(i, self.evidence) for i in self.case.b]},
        }
        if self.reference_variant == "instance":
            payload["instance_reference"] = self.case.reference.get("instance", [])
        elif self.reference_variant == "invariant":
            payload["invariant_reference"] = self.case.reference.get("invariant", [])
        return json.dumps(payload, indent=2)

    def _system(self, placement: str) -> str:
        # both_inert has no live side, so no tools regardless of interrogation.
        tools_note = CLOSER_NOTOOLS if placement == "both_inert" else _tools_note(self.interrogation)
        return (SYSTEM_PREAMBLE + tools_note + PLACEMENT_NOTE.get(placement, "")
                + REFERENCE_NOTE.get(self.reference_variant, ""))

    def _dispatch(self, oracle: Oracle, name: str, args: dict):
        if name == "interrogate":
            # In "lookup" mode the agent must name an attribute: no discovery. Refuse a bare
            # interrogate, and suppress the oracle's available-attribute hint (itself a discovery
            # interrogation) so the reach axis stays clean.
            attribute = args.get("attribute")
            if self.interrogation == "lookup" and not attribute:
                return {"ok": False, "answer": {},
                        "message": "interrogate requires an attribute name in this mode"}
            r = oracle.interrogate(args.get("individual_id", ""), attribute)
            message = r.message
            if self.interrogation == "lookup" and "interrogable attributes:" in message:
                message = message.split(";")[0].strip()
            return {"ok": r.ok, "answer": r.answer, "message": message}
        if name == "virtual_provision":
            r = oracle.virtual_provision(args.get("a_id", ""), args.get("b_id", ""))
            return {"ok": r.ok, "answer": r.answer, "message": r.message}
        return {"ok": False, "message": f"unknown tool '{name}'"}

    def reconcile(self, placement: str = "both_cognitive") -> InstanceReconciliation:
        client = self._get_client()
        oracle = Oracle(self.case, placement, inert_side=self.inert_side, budget=self.budget)
        rec = InstanceReconciliation(placement=placement, reference_variant=self.reference_variant)
        input_list = [{"role": "system", "content": self._system(placement)},
                      {"role": "user", "content": self._user_payload()}]
        rec.transcript.append({"step": "prompt", "placement": placement,
                               "reference_variant": self.reference_variant,
                               "evidence": sorted(self.evidence),
                               "interrogation": self.interrogation})
        tok = {"prompt": 0, "completion": 0, "total": 0, "reasoning": 0}
        tools = SUBMIT_ONLY if placement == "both_inert" else _tools_for(self.interrogation)
        t0 = time.time()
        submitted_args = None
        turns = 0

        for turn in range(self.max_turns):
            turns = turn + 1
            resp = client.responses.create(model=self.model, input=input_list,
                                            tools=tools, tool_choice="auto")
            self._accumulate_usage(resp, tok)
            out = list(_g(resp, "output", []) or [])
            input_list += out                       # feed reasoning + calls back for continuation
            calls = [it for it in out if _g(it, "type") == "function_call"]
            for it in out:
                if _g(it, "type") == "message":
                    txt = self._message_text(it)
                    if txt:
                        rec.transcript.append({"step": "assistant_text", "text": txt})
            if not calls:
                input_list.append({"role": "user", "content":
                                   "If you are done, call submit_alignment with your final answer."})
                if turn >= 1:
                    break
                continue
            done = False
            for call in calls:
                name = _g(call, "name")
                cid = _g(call, "call_id") or _g(call, "id") or "call_0"
                args = self._parse_args(_g(call, "arguments", "{}"))
                if name == "submit_alignment":
                    submitted_args = args
                    rec.transcript.append({"step": "submit", "args": args})
                    input_list.append({"type": "function_call_output", "call_id": cid,
                                       "output": json.dumps({"ok": True, "received": True})})
                    done = True
                    continue
                result = self._dispatch(oracle, name, args)
                rec.transcript.append({"step": name, "args": args, "result": result})
                input_list.append({"type": "function_call_output", "call_id": cid,
                                   "output": json.dumps(result)})
            if done:
                break

        rec.effort = {
            "prompt_tokens": tok["prompt"], "completion_tokens": tok["completion"],
            "total_tokens": tok["total"], "reasoning_tokens": tok["reasoning"],
            "latency_s": round(time.time() - t0, 2), "model": self.model, "turns": turns,
        }
        self._finalize(rec, oracle, submitted_args)
        return rec

    def _finalize(self, rec, oracle, args):
        a_ids, b_ids = set(self.case.a_by_id), set(self.case.b_by_id)
        matched_a, matched_b = set(), set()
        if args:
            rec.submitted = True
            for c in args.get("correspondences", []):
                aid, bid = c.get("a_id"), c.get("b_id")
                if aid in a_ids and bid in b_ids:
                    pair = frozenset((aid, bid))
                    rec.proposed.append(pair)
                    rec.confidence[pair] = float(c.get("confidence", 0.0))
                    matched_a.add(aid)
                    matched_b.add(bid)
        rec.residual_a = [i for i in a_ids if i not in matched_a]
        rec.residual_b = [i for i in b_ids if i not in matched_b]
        rec.confirmed_pairs = set(oracle.confirmed_pairs)
        rec.oracle_calls = dict(oracle.calls)
        rec.interrogated = set(oracle.interrogated)

    # -- SDK/​fake-tolerant plumbing ----------------------------------------------------
    def _parse_args(self, raw):
        try:
            return json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (ValueError, TypeError):
            return {}

    def _message_text(self, item) -> str:
        content = _g(item, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                t = _g(c, "text")
                if t:
                    parts.append(t)
            return " ".join(parts)
        return ""

    def _accumulate_usage(self, resp, tok):
        usage = _g(resp, "usage")
        if not usage:
            return
        tok["prompt"] += _g(usage, "input_tokens", 0) or 0
        tok["completion"] += _g(usage, "output_tokens", 0) or 0
        tok["total"] += _g(usage, "total_tokens", 0) or 0
        details = _g(usage, "output_tokens_details")
        if details:
            tok["reasoning"] += _g(details, "reasoning_tokens", 0) or 0
