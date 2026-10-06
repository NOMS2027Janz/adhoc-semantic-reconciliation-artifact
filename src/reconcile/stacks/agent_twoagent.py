"""Two-agent negotiation stack: two separate agents reconcile under real information
asymmetry, instead of one model shown both lifted models.

This is the honest form of the fully-cognitive case. Agent A holds ONLY model A; Agent B
holds ONLY model B. Neither sees the other's concepts except:

  * a public catalogue each agent advertises for its own side, carrying label and kind only
    (the surface, so a false cognate can still be proposed and must be caught by meaning); and
  * what the other agent volunteers in answers to questions (gloss, example, relations).

The two agents alternate turns. On its turn an agent may:
  * answer the other's outstanding questions about its own concepts, from its private model;
  * ask about the other's concepts when it needs their meaning to decide;
  * propose correspondences (my concept <-> one of yours) with a confidence and rationale;
  * ratify or reject proposals the other made about MY concepts (each side is the authority
    on its own side -- this is the bilateral confirmation Part I describes); and
  * mark its own concepts that it believes have no counterpart.

A correspondence is CONFIRMED only when one side proposes it and the owning side ratifies it,
or when both sides propose it independently. That is the closed loop: neither side alone
decides a cross-model correspondence.

Placements:
  * both_cognitive -> the negotiation above.
  * one_inert / both_inert -> there is no partner to negotiate with (a mute side cannot answer
    or ratify), so these delegate to the single-agent OpenAIAgentStack, which reconstructs the
    inert side(s) from structure and instances. Negotiation is a both-live phenomenon by design.

Effort is summed across every agent turn (both sides, all rounds), so the cost of the exchange
is measured, not hidden. Requires `pip install openai` and OPENAI_API_KEY, except with an
injected client (used by the offline test).
"""
from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, Field

from reconcile.reference import Reference
from reconcile.stacks.base import ReasoningStack, Reconciliation
from reconcile.stacks.agent_openai import OpenAIAgentStack

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6")
DEFAULT_MAX_ROUNDS = 6  # one round = A then B; the loop stops early when both declare done


# ---- the structured turn an agent returns ------------------------------------------------
class Answer(BaseModel):
    my_id: str
    gloss: str = ""
    example: str = ""
    relations: str = ""


class Question(BaseModel):
    their_id: str
    ask: str


class Proposal(BaseModel):
    my_id: str
    their_id: str
    confidence: float
    rationale: str


class Ratification(BaseModel):
    their_id: str   # the proposer's concept id
    my_id: str      # my concept the proposer paired it with
    accept: bool
    reason: str


class Experiment(BaseModel):
    my_id: str      # your concept
    their_id: str   # one of the other's concepts to provision it against
    question: str = ""


class Turn(BaseModel):
    answers: list[Answer]
    questions: list[Question]
    proposals: list[Proposal]
    ratifications: list[Ratification]
    experiments: list[Experiment] = Field(default_factory=list)
    no_counterpart: list[str]
    done: bool


def _system(side: str, other: str, use_reference: bool, has_oracle: bool = False,
            has_frame: bool = False) -> str:
    s = (
        f"You are Agent {side}, the sole authority on model {side} of ONE network. You are "
        f"reconciling with Agent {other}, who holds model {other}. You see your OWN model in "
        f"full; you learn about {other}'s concepts only from its advertised catalogue (label and "
        "kind) and from its answers to your questions. Reconcile by MEANING, not by label: two "
        "concepts with similar names may denote different things (a false cognate), and two with "
        "different names may denote the same thing.\n\n"
        "Each turn, return: answers to the other's outstanding questions about YOUR concepts "
        "(give gloss, example, and a short relations summary from your model); questions about the "
        f"other's concepts when you need their meaning; proposals (your concept id <-> one of "
        f"{other}'s ids) with a confidence in [0,1] and a rationale; ratifications of proposals the "
        "OTHER made about YOUR concepts (accept only if the other's concept truly denotes the same "
        "thing yours does -- you are the authority on your side); and the ids of your concepts you "
        "believe have no counterpart. Set done=true when you have nothing further to add. "
        "Propose a correspondence at most once, and use only ids that exist (yours, and the other's "
        "advertised ids). Before you set done=true, make sure you have proposed a correspondence for "
        "every one of your concepts that has a plausible counterpart in the other's catalogue, asked "
        "about any of the other's concepts you still cannot place, and ratified (accept or reject) "
        "every outstanding proposal the other has made about your concepts. Set done=true only when "
        "nothing remains to answer, ask, propose, or ratify."
    )
    if use_reference:
        s += (
            "\n\nA shared reference is in play: your concepts carry a 'ref' entry id where their "
            "author bound them. If you and the other can see you are bound to the SAME reference "
            "entry, that settles the correspondence directly; two concepts bound to DIFFERENT "
            "entries do not correspond even if their labels look alike."
        )
    if has_frame:
        s += (
            "\n\nA SHARED REFERENCE FRAME is given ('shared_frame'): a compact set of identity-only "
            "entries (id, label, definition, example) that you and the other agent jointly constructed "
            "from both models before this negotiation. It is your common anchor. Bind each of YOUR "
            "concepts to the entry whose definition and example fit it, matching on MEANING rather than "
            "surface label. When you can see that one of your concepts and one of the other's plausibly "
            "bind to the SAME entry, propose that correspondence even if the two concepts sit at "
            "different layers or carry different local names; the shared entry is the frame that makes "
            "them correspond. An entry that only shares a surface word with your concept is not a match."
        )
    if has_oracle:
        s += (
            "\n\nYou may run a DECISIVE VIRTUAL EXPERIMENT on any candidate correspondence: list it "
            "under 'experiments' (your concept id and one of the other's ids). The shared virtual "
            "network will PROVISION that pairing, operate it, and read back whether the invariants a "
            "correct translation must preserve (endpoint identity, capacity, connectivity, layer "
            "relationships) hold. The verdict -- confirmed (same thing), or refuted (a false cognate, "
            "or no counterpart) -- is returned to BOTH of you before the next turn and is DECISIVE: "
            "trust it over surface judgement. Use experiments to settle any correspondence you are "
            "unsure of and to test suspected false cognates, rather than deferring them. Experiments "
            "are limited, so spend them on the pairs that matter; a correspondence the experiment "
            "confirms is settled without further ratification."
        )
    return s


def _private_view(model, use_reference: bool) -> list[dict]:
    out = []
    for c in model.concepts:
        d = {"id": c.id, "label": c.label, "kind": c.kind, "synonyms": list(c.synonyms),
             "gloss": c.gloss, "example": c.example, "relations": list(c.relations),
             "instances": list(c.instances)}
        if use_reference:
            d["ref"] = c.ref
        out.append(d)
    return out


def _catalogue(model) -> list[dict]:
    """The public advertisement: surface only (id, label, kind)."""
    return [{"id": c.id, "label": c.label, "kind": c.kind} for c in model.concepts]


def _effort_from_usage(usage) -> dict:
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
    }


class TwoAgentStack(ReasoningStack):
    """Two agents negotiating a reconciliation under information asymmetry."""

    def __init__(self, use_reference: bool = False, model: str | None = None,
                 inert_side: str = "b", client=None, max_rounds: int = DEFAULT_MAX_ROUNDS,
                 oracle=None, shared_frame=None):
        self.uses_reference = use_reference
        self.model = model or DEFAULT_MODEL
        self.inert_side = inert_side
        self.max_rounds = max_rounds
        self._client = client
        self.oracle = oracle   # a SchemaOracle enables the decisive virtual experiment
        # a constructed shared frame (a Reference): the candidate-surfacing act, injected into
        # BOTH agents so they bind their own side to it in-loop. Distinct from use_reference,
        # which exposes each concept's own AUTHORED ref binding.
        self.shared_frame = shared_frame
        tag = (("+frame" if shared_frame is not None else "")
               + ("+exp" if oracle is not None else "")
               + ("/ref" if use_reference else "/no-ref"))
        self.name = f"two-agent{tag}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _turn(self, client, side, other, private, their_catalogue, transcript):
        system = _system(side, other, self.uses_reference, has_oracle=self.oracle is not None,
                         has_frame=self.shared_frame is not None)
        payload = {
            "your_model": private,
            f"agent_{other}_catalogue": their_catalogue,
            "transcript_so_far": transcript,
        }
        if self.shared_frame is not None:
            payload["shared_frame"] = [
                {"id": e.id, "label": e.label, "class": e.cls,
                 "definition": e.definition, "example": e.example,
                 "synonyms": list(e.synonyms)} for e in self.shared_frame.entries]
        user = json.dumps(payload, indent=2)
        t0 = time.time()
        completion = client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format=Turn,
        )
        elapsed = time.time() - t0
        turn = completion.choices[0].message.parsed
        eff = _effort_from_usage(getattr(completion, "usage", None))
        eff["latency_s"] = round(elapsed, 2)
        return turn, eff

    def _negotiate(self, a, b, reference) -> Reconciliation:
        client = self._get_client()
        cat_a, cat_b = _catalogue(a), _catalogue(b)
        priv_a = _private_view(a, self.uses_reference)
        priv_b = _private_view(b, self.uses_reference)
        transcript: list[dict] = []
        # proposals seen as directed (proposer_side, frozenset(pair)); accepts / rejects by pair
        proposed_by: dict[frozenset, set] = {}
        proposed_conf: dict[frozenset, float] = {}   # pair -> best proposer confidence seen
        accepts: set[frozenset] = set()
        rejects: set[frozenset] = set()
        oracle_confirmed: set[frozenset] = set()   # decisive-experiment verdicts
        oracle_refuted: set[frozenset] = set()
        a_ids, b_ids = set(a.by_id), set(b.by_id)
        total = {"total_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0}
        turns = 0

        def record(side, turn: Turn):
            # proposals: my_id is on `side`, their_id on the other side
            for p in turn.proposals:
                if side == "A":
                    pair = frozenset((p.my_id, p.their_id)) if p.my_id in a_ids and p.their_id in b_ids else None
                else:
                    pair = frozenset((p.their_id, p.my_id)) if p.my_id in b_ids and p.their_id in a_ids else None
                if pair:
                    proposed_by.setdefault(pair, set()).add(side)
                    proposed_conf[pair] = max(proposed_conf.get(pair, 0.0),
                                              float(getattr(p, "confidence", 0.0) or 0.0))
            # ratifications: their_id is the proposer's (other side), my_id is mine (`side`)
            for r in turn.ratifications:
                if side == "A":
                    pair = frozenset((r.my_id, r.their_id)) if r.my_id in a_ids and r.their_id in b_ids else None
                else:
                    pair = frozenset((r.their_id, r.my_id)) if r.my_id in b_ids and r.their_id in a_ids else None
                if pair:
                    (accepts if r.accept else rejects).add(pair)

        last_done = {"A": False, "B": False}
        for _ in range(self.max_rounds):
            for side, other, private, their_cat in (
                ("A", "B", priv_a, cat_b), ("B", "A", priv_b, cat_a)):
                turn, eff = self._turn(client, side, other, private, their_cat, transcript)
                turns += 1
                total["total_tokens"] += eff["total_tokens"]
                total["reasoning_tokens"] += eff["reasoning_tokens"]
                total["latency_s"] += eff["latency_s"]
                record(side, turn)
                exp_results = []
                if self.oracle is not None:
                    for e in turn.experiments:
                        res = self.oracle.virtual_provision(e.my_id, e.their_id)
                        exp_results.append({"a": res.a_id, "b": res.b_id, "ok": res.ok,
                                            "confirmed": res.confirmed, "relation": res.relation})
                        if not res.ok:
                            continue
                        pair = (frozenset((res.a_id, res.b_id))
                                if res.a_id in a_ids and res.b_id in b_ids else None)
                        if pair:
                            (oracle_confirmed if res.confirmed else oracle_refuted).add(pair)
                entry = {"from": side, **turn.model_dump()}
                if exp_results:
                    entry["experiment_results"] = exp_results
                transcript.append(entry)
                last_done[side] = turn.done
            if last_done["A"] and last_done["B"]:
                break

        # confirmed: proposed AND (ratified accept by the owner, or proposed by both sides),
        # and never rejected.
        self.last_transcript = transcript   # exposed for the transcript diagnostic
        # dialogue confirms (propose + owner-ratify, or mutual propose); then the decisive
        # experiment overrides: a confirmed verdict binds, a refuted verdict removes.
        confirmed = {
            pair for pair, sides in proposed_by.items()
            if (pair in accepts or len(sides) == 2) and pair not in rejects
        }
        confirmed |= oracle_confirmed
        confirmed -= oracle_refuted
        matched_a = {next(i for i in pair if i in a_ids) for pair in confirmed}
        matched_b = {next(i for i in pair if i in b_ids) for pair in confirmed}
        total["latency_s"] = round(total["latency_s"], 2)
        total["model"] = self.model
        return Reconciliation(
            stack=self.name, uses_reference=self.uses_reference, placement="both_cognitive",
            proposed=list(confirmed),
            residual_a=[c.id for c in a.concepts if c.id not in matched_a],
            residual_b=[c.id for c in b.concepts if c.id not in matched_b],
            work={"candidates": len(confirmed), "turns": turns,
                  "bilateral_checks": len(accepts) + len(rejects),
                  "experiments": (self.oracle.calls if self.oracle is not None else 0)},
            effort=total,
            # a decisive-experiment verdict binds at full confidence; otherwise the best
            # proposer confidence the dialogue attached to the pair.
            confidence={pair: (1.0 if pair in oracle_confirmed else proposed_conf.get(pair, 0.5))
                        for pair in confirmed},
        )

    def reconcile(self, a, b, reference: Reference | None = None,
                  placement: str = "both_cognitive") -> Reconciliation:
        if placement == "both_cognitive":
            return self._negotiate(a, b, reference)
        # no partner to negotiate with: reconstruct the mute side(s) with the single agent.
        single = OpenAIAgentStack(use_reference=self.uses_reference, model=self.model,
                                  inert_side=self.inert_side, client=self._client)
        rec = single.reconcile(a, b, reference=reference, placement=placement)
        rec.stack = self.name + "/single-inert"
        return rec
