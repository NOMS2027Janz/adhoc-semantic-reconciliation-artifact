"""Intent-reconciliation agent: a phase-aware bounded tool-use loop over the IntentOracle.

Like the instance stack, this runs a loop in which the oracle (see ``reconcile.intent
.IntentOracle``) is exposed to the model as tools, and the model calls them until it
submits or the turn budget is spent. It serves the three cognitive phases of the setting
with one loop, selected by ``phase``:

  * ``refine``    — for each intent, pick a realisation and say whether it satisfies every
                    bound (refine-down + satisfaction). Tool: check_feasibility.
  * ``negotiate`` — for each infeasible intent, obtain the best-achievable offer and decide
                    accept/reject against the movable policy in force. Tools: best_achievable,
                    check_feasibility, consult_policy.
  * ``assure``    — walk one multi-hop lifecycle trajectory: per hop, the fulfilment status
                    and the renegotiation decision. Tools: read_operational, check_feasibility,
                    best_achievable, consult_policy.

It uses the OpenAI Responses API (reasoning items fed back each turn), and the tools'
availability follows the cognition placement, which the oracle enforces. The whole exchange
is captured as a transcript, so a run — especially a lifecycle set-piece — can be shown, not
only scored. The client is injectable, so the loop can be driven by a scripted fake, no API.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from reconcile.intent import IntentCase, IntentOracle


@dataclass
class IntentReconciliation:
    phase: str
    placement: str
    reference_variant: str
    policy_id: str = ""
    trajectory_id: str = ""
    refinements: list = field(default_factory=list)     # [{intent_id, realisation_id, satisfies}]
    decisions: list = field(default_factory=list)        # [{intent_id, offer_realisation, decision}]
    hops: list = field(default_factory=list)             # [{hop_id, fulfilment, decision, new_realisation}]
    oracle_calls: dict = field(default_factory=dict)
    effort: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    submitted: bool = False


SYSTEM_COMMON = (
    "You reconcile a declarative INTENT against a concrete network REALISATION. The reconciliation "
    "is a REFINEMENT, not an equivalence: a bound (bandwidth >=, latency <=, availability >=, "
    "protection required) is not equal to a realisation, it is SATISFIED BY one. Agent O states the "
    "intent's bounds; Agent N offers realisations from a catalogue, each with advertised attributes "
    "(bandwidth in Gbit/s, latency in ms, availability as a fraction, protection 'none'|'1+1', and a "
    "cost). The advertised catalogue can differ from what a path ACTUALLY delivers right now; where a "
    "side is live you can check the real figures with the tools. Never conflate an expectation "
    "('latency <= 5ms') with a measured metric ('latency = 6ms'); scale bandwidth bounds against the "
    "discrete ODU rates correctly. Where a decision cannot be settled from what you have, REFER it "
    "onward rather than guess.\n\n"
)

PHASE_SYSTEM = {
    "refine": (
        "TASK (refine-down + satisfaction): for each intent, choose the realisation you would "
        "provision and say whether it satisfies EVERY bound. Prefer a realisation you have confirmed "
        "over one you assume. If no realisation satisfies every bound, set realisation_id to \"\" and "
        "satisfies=false (the intent goes to negotiation). Tool:\n"
        " - check_feasibility(intent_id, realisation_id): the provider reports, live, which bounds a "
        "realisation ACTUALLY satisfies now. Use it wherever advertised evidence might mislead.\n"
        "Call submit_refinement exactly once when done, covering every intent."),
    "negotiate": (
        "TASK (feasibility + the judge): each intent here has no realisation that meets every bound. "
        "For each, obtain the BEST-ACHIEVABLE offer (the realisation that degrades only the "
        "least-important bounds under the priority order) and DECIDE accept or reject against the "
        "movable policy in force — its hard (must-hold) bounds and affordability floor. Tools:\n"
        " - best_achievable(intent_id): the provider computes the best-achievable offer.\n"
        " - check_feasibility(intent_id, realisation_id): the actual bounds a realisation satisfies.\n"
        " - consult_policy(intent_id, realisation_id): the consumer's movable policy renders "
        "accept/reject on an offer. If it is unavailable (a mute consumer), REFER the decision.\n"
        "Call submit_negotiation exactly once when done, covering every intent."),
    "assure": (
        "TASK (assure-up over a service lifecycle): walk the trajectory hop by hop. At each hop you "
        "are given the bounds in force (the current agreed SLA) and an event. Read the operational "
        "state, classify fulfilment as 'met', 'at_risk', or 'breach', and decide what to do: 'hold' "
        "(no change), 'accept' (migrate to a new realisation the policy accepts), or 'refer' (a "
        "decision the consumer must make). Carry the in-service realisation forward from hop to hop. "
        "Tools:\n"
        " - read_operational(hop_id): live telemetry — the in-service realisation and its reading.\n"
        " - check_feasibility / best_achievable / consult_policy: as in negotiation, to find and "
        "judge a remediation offer on a breach.\n"
        "Call submit_lifecycle exactly once when done, covering every hop in order."),
}

PLACEMENT_NOTE = {
    "both_cognitive": "\n\nBOTH sides are live: the provider computes live feasibility and "
                      "best-achievable offers, and the consumer's judgement (or its movable policy) is "
                      "available.",
    "one_inert": "\n\nONE side is INERT — a static description. On an inert PROVIDER you have only the "
                 "advertised catalogue (no live feasibility, no fresh best-achievable, no telemetry); "
                 "on an inert CONSUMER its movable policy may still decide, or, if it is a mute "
                 "description, you must refer decisions onward.",
    "both_inert": "\n\nBOTH sides are INERT. No live feasibility, no best-achievable, no policy: "
                  "propose from advertised evidence only and refer every acceptance decision onward.",
}

REFERENCE_NOTE = {
    "none": "",
    "unit": "\n\nYou are given a UNIT / VALUE-SET reference: it pins what each unit means, the discrete "
            "value hierarchies (ODU rates, protection classes, availability nines), and the separation "
            "of expectation-kind bounds from realisation-kind values. Use it to avoid unit-scale and "
            "expectation-vs-metric mistakes.",
    "invariant": "\n\nYou are given an INVARIANT reference: the committed guarantee floor published per "
                 "realisation. Use it to check satisfaction against the published guarantees when a side "
                 "is inert and cannot be probed.",
}


def _fn_tool(name, description, properties, required):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


T_CHECK = _fn_tool("check_feasibility",
                   "Ask the provider which bounds a realisation ACTUALLY satisfies right now.",
                   {"intent_id": {"type": "string"}, "realisation_id": {"type": "string"}},
                   ["intent_id", "realisation_id"])
T_BEST = _fn_tool("best_achievable",
                  "Ask the provider for the best-achievable offer for an intent under the priority order.",
                  {"intent_id": {"type": "string"}}, ["intent_id"])
T_POLICY = _fn_tool("consult_policy",
                    "Ask the consumer's movable policy to accept or reject an offer for an intent.",
                    {"intent_id": {"type": "string"}, "realisation_id": {"type": "string"}},
                    ["intent_id", "realisation_id"])
T_OPER = _fn_tool("read_operational",
                  "Read live telemetry: the in-service realisation and its reading at a lifecycle hop.",
                  {"hop_id": {"type": "string"}}, ["hop_id"])

T_SUBMIT_REFINE = _fn_tool(
    "submit_refinement", "Submit the refine-down result for every intent.",
    {"refinements": {"type": "array", "items": {"type": "object", "properties": {
        "intent_id": {"type": "string"}, "realisation_id": {"type": "string"},
        "satisfies": {"type": "boolean"}},
        "required": ["intent_id", "realisation_id", "satisfies"], "additionalProperties": False}}},
    ["refinements"])
T_SUBMIT_NEG = _fn_tool(
    "submit_negotiation", "Submit the negotiation decision for every intent.",
    {"decisions": {"type": "array", "items": {"type": "object", "properties": {
        "intent_id": {"type": "string"}, "offer_realisation": {"type": "string"},
        "decision": {"type": "string", "enum": ["accept", "reject", "refer"]}},
        "required": ["intent_id", "offer_realisation", "decision"], "additionalProperties": False}}},
    ["decisions"])
T_SUBMIT_LIFE = _fn_tool(
    "submit_lifecycle", "Submit the fulfilment and decision for every hop, in order.",
    {"hops": {"type": "array", "items": {"type": "object", "properties": {
        "hop_id": {"type": "string"},
        "fulfilment": {"type": "string", "enum": ["met", "at_risk", "breach"]},
        "decision": {"type": "string", "enum": ["hold", "accept", "refer"]},
        "new_realisation": {"type": "string"}},
        "required": ["hop_id", "fulfilment", "decision", "new_realisation"],
        "additionalProperties": False}}},
    ["hops"])

PHASE_TOOLS = {
    "refine": [T_CHECK, T_SUBMIT_REFINE],
    "negotiate": [T_BEST, T_CHECK, T_POLICY, T_SUBMIT_NEG],
    "assure": [T_OPER, T_CHECK, T_BEST, T_POLICY, T_SUBMIT_LIFE],
}
SUBMIT_NAME = {"refine": "submit_refinement", "negotiate": "submit_negotiation",
               "assure": "submit_lifecycle"}


def _g(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class IntentAgentStack:
    def __init__(self, case: IntentCase, model: str, phase: str, reference_variant: str = "none",
                 policy_id: str = "", trajectory_id: str = "", inert_side: str = "n",
                 consumer_mode: str = "policy", budget: int | None = 30,
                 max_turns: int | None = None, client=None):
        self.case = case
        self.model = model
        self.phase = phase
        self.reference_variant = reference_variant
        self.policy_id = policy_id
        self.trajectory_id = trajectory_id
        self.inert_side = inert_side
        self.consumer_mode = consumer_mode
        self.budget = budget
        self.max_turns = max_turns or max(12, (budget if budget is not None else 18) + 6)
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _policy(self) -> dict:
        return self.case.policy_by_id.get(self.policy_id, {})

    def _reference_payload(self):
        if self.reference_variant == "unit":
            return {"unit_value_set_reference": self.case.reference.get("unit_value_set", [])}
        if self.reference_variant == "invariant":
            return {"invariant_reference": self.case.reference.get("invariant", [])}
        return {}

    def _catalogue_view(self):
        return [{"id": r["id"], "pair": r["pair"], "path": r["path"], "capacity": r["capacity"],
                 "advertised": r["advertised"]} for r in self.case.realisations]

    def _user_payload(self) -> str:
        cat = self._catalogue_view()
        if self.phase == "refine":
            payload = {"task": "refine-down + satisfaction",
                       "intents": [{"id": i["id"], "pair": i["pair"], "flow_class": i["flow_class"],
                                    "bounds": i["bounds"]} for i in self.case.intents],
                       "catalogue": cat}
        elif self.phase == "negotiate":
            neg = self.case.gold.get("negotiation_intents") or [i["id"] for i in self.case.intents]
            payload = {"task": "feasibility + judge (negotiation)",
                       "policy_in_force": self._policy(),
                       "intents": [{"id": i["id"], "pair": i["pair"], "flow_class": i["flow_class"],
                                    "bounds": i["bounds"]} for i in self.case.intents if i["id"] in neg],
                       "catalogue": cat}
        else:  # assure
            traj = next(t for t in self.case.lifecycle if t["id"] == self.trajectory_id)
            intent = self.case.intent_by_id[traj["intent_id"]]
            payload = {"task": "assure-up over a lifecycle",
                       "trajectory": traj["id"], "narrative": traj.get("narrative", ""),
                       "intent": {"id": intent["id"], "pair": intent["pair"],
                                  "flow_class": intent["flow_class"]},
                       "policy_in_force": self._policy(),
                       "initial_realisation": traj["initial_realisation"],
                       "hops": [{"hop_id": h["hop_id"], "origin": h["origin"], "kind": h["kind"],
                                 "bounds_in_force": h["bounds"], "event": h["event"],
                                 **({"offered_target": h["target"]} if "target" in h else {})}
                                for h in traj["hops"]],
                       "catalogue": cat}
        payload.update(self._reference_payload())
        return json.dumps(payload, indent=2)

    def _system(self, placement: str) -> str:
        return (SYSTEM_COMMON + PHASE_SYSTEM[self.phase] + PLACEMENT_NOTE.get(placement, "")
                + REFERENCE_NOTE.get(self.reference_variant, ""))

    def _dispatch(self, oracle: IntentOracle, name: str, args: dict):
        if name == "check_feasibility":
            r = oracle.check_feasibility(args.get("intent_id", ""), args.get("realisation_id", ""))
        elif name == "best_achievable":
            r = oracle.best_achievable(args.get("intent_id", ""))
        elif name == "consult_policy":
            r = oracle.consult_policy(args.get("intent_id", ""), args.get("realisation_id", ""))
        elif name == "read_operational":
            r = oracle.read_operational(self.trajectory_id, args.get("hop_id", ""))
        else:
            return {"ok": False, "message": f"unknown tool '{name}'"}
        return {"ok": r.ok, "answer": r.answer, "message": r.message}

    def reconcile(self, placement: str = "both_cognitive") -> IntentReconciliation:
        client = self._get_client()
        oracle = IntentOracle(self.case, placement, policy=self._policy(),
                              inert_side=self.inert_side, consumer_mode=self.consumer_mode,
                              budget=self.budget)
        rec = IntentReconciliation(phase=self.phase, placement=placement,
                                   reference_variant=self.reference_variant,
                                   policy_id=self.policy_id, trajectory_id=self.trajectory_id)
        input_list = [{"role": "system", "content": self._system(placement)},
                      {"role": "user", "content": self._user_payload()}]
        rec.transcript.append({"step": "prompt", "phase": self.phase, "placement": placement,
                               "reference_variant": self.reference_variant,
                               "policy_id": self.policy_id, "trajectory_id": self.trajectory_id})
        tok = {"prompt": 0, "completion": 0, "total": 0, "reasoning": 0}
        tools = PHASE_TOOLS[self.phase]
        submit_name = SUBMIT_NAME[self.phase]
        t0 = time.time()
        submitted_args = None
        turns = 0

        for turn in range(self.max_turns):
            turns = turn + 1
            resp = client.responses.create(model=self.model, input=input_list,
                                            tools=tools, tool_choice="auto")
            self._accumulate_usage(resp, tok)
            out = list(_g(resp, "output", []) or [])
            input_list += out
            calls = [it for it in out if _g(it, "type") == "function_call"]
            for it in out:
                if _g(it, "type") == "message":
                    txt = self._message_text(it)
                    if txt:
                        rec.transcript.append({"step": "assistant_text", "text": txt})
            if not calls:
                input_list.append({"role": "user", "content":
                                   f"If you are done, call {submit_name} with your final answer."})
                if turn >= 1:
                    break
                continue
            done = False
            for call in calls:
                name = _g(call, "name")
                cid = _g(call, "call_id") or _g(call, "id") or "call_0"
                cargs = self._parse_args(_g(call, "arguments", "{}"))
                if name == submit_name:
                    submitted_args = cargs
                    rec.transcript.append({"step": "submit", "args": cargs})
                    input_list.append({"type": "function_call_output", "call_id": cid,
                                       "output": json.dumps({"ok": True, "received": True})})
                    done = True
                    continue
                result = self._dispatch(oracle, name, cargs)
                rec.transcript.append({"step": name, "args": cargs, "result": result})
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
        rec.oracle_calls = dict(oracle.calls)
        if not args:
            return
        rec.submitted = True
        if self.phase == "refine":
            rec.refinements = args.get("refinements", [])
        elif self.phase == "negotiate":
            rec.decisions = args.get("decisions", [])
        else:
            rec.hops = args.get("hops", [])

    # -- SDK/fake-tolerant plumbing --------------------------------------------------------
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
            parts = [t for c in content if (t := _g(c, "text"))]
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
