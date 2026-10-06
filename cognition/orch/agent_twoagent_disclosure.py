"""Disclosure-enabled instrumented two-agent stack.

Adds the opening DISCLOSURE step the reconciliation protocol was missing. Before any
negotiation turn, each side advertises the system it manages and the dialect its model is
written in -- a published standard (e.g. ONF TAPI, IETF TEAS) or a bespoke vocabulary --
together with its catalogue of concept ids and labels. The two advertisements are recorded
as the first entries in the transcript, so the handshake is visible in the trace, and each
agent is told in its system prompt what the other disclosed.

This makes legitimate the fact that an agent knows the other's catalogue at turn 0: the
catalogue was advertised on the basis of a disclosed, named provenance, not assumed. Only the
SURFACE is advertised (id, label, kind, plus the system/dialect); private semantics (gloss,
example, relations) are still withheld and must be elicited, exactly as before.

Shelf code for the cognition note. Subclasses the instrumented stack; the base stack in
src/ is untouched. The advertisement content is built from the models' own declared
``system``/``dialect`` metadata -- it is disclosed fact, not model-generated text.
"""
from __future__ import annotations

import json
import sys
import time

from reconcile.stacks.agent_twoagent import (
    Turn,
    _system,
    _effort_from_usage,
    _catalogue,
    _private_view,
)
from reconcile.stacks.base import Reconciliation
from agent_twoagent_instrumented import TwoAgentStackInstrumented, TurnI, _PLAN_INSTR


def _disclosure_clause(own: tuple[str, str], other: tuple[str, str]) -> str:
    return (
        "\n\nPROVENANCE DISCLOSURE (already exchanged at the outset). Each side has advertised the "
        "system it manages and the dialect its model is written in, together with its catalogue of "
        f"concept ids and labels. You manage system {own[0]!r}, modelled in dialect {own[1]!r}. The "
        f"other side manages system {other[0]!r}, modelled in dialect {other[1]!r}. These two "
        "advertisements (system, dialect, catalogue) are the first entries in the transcript. Only "
        "the surface was advertised; neither side disclosed its private glosses, examples, or "
        "relations, which you must still elicit. Use the disclosed dialects as context -- two "
        "different dialects may name the same thing differently, or reuse the same word for "
        "different things -- but reconcile by MEANING, never by the dialect or label alone."
    )


class TwoAgentStackDisclosure(TwoAgentStackInstrumented):
    """Instrumented two-agent stack preceded by a provenance-disclosure handshake."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.name + "+disclose"
        self._prov: dict[str, tuple[str, str]] = {}

    # -- per-turn call: add the other side's disclosed provenance to the payload and the
    #    disclosure clause to the system prompt. Otherwise identical to the instrumented turn.
    def _turn(self, client, side, other, private, their_catalogue, transcript):
        own = self._prov.get(side, ("", ""))
        oth = self._prov.get(other, ("", ""))
        system = _system(
            side, other, self.uses_reference,
            has_oracle=self.oracle is not None,
            has_frame=self.shared_frame is not None,
        ) + _PLAN_INSTR + _disclosure_clause(own, oth)
        payload = {
            "your_model": private,
            f"agent_{other}_disclosed": {"system": oth[0], "dialect": oth[1]},
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
            response_format=TurnI,
        )
        elapsed = time.time() - t0
        turn = completion.choices[0].message.parsed
        eff = _effort_from_usage(getattr(completion, "usage", None))
        eff["latency_s"] = round(elapsed, 2)
        try:
            print(f"  [{side}] turn done in {eff['latency_s']}s "
                  f"({eff['total_tokens']} tok) -- decision={turn.plan.decision!r} "
                  f"closeness={turn.plan.closeness} "
                  f"(q={len(turn.questions)}, prop={len(turn.proposals)}, "
                  f"ratif={len(turn.ratifications)}, done={turn.done})",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
        return turn, eff

    # -- negotiation: faithful copy of TwoAgentStack._negotiate, with two additions:
    #    (1) record each side's declared provenance in self._prov; (2) seed the transcript with
    #    the two opening advertisements so the handshake is part of the trace.
    def _negotiate(self, a, b, reference) -> Reconciliation:
        client = self._get_client()
        cat_a, cat_b = _catalogue(a), _catalogue(b)
        priv_a = _private_view(a, self.uses_reference)
        priv_b = _private_view(b, self.uses_reference)

        # (1) declared provenance, for the per-turn disclosure framing
        self._prov = {"A": (a.system, a.dialect), "B": (b.system, b.dialect)}

        # (2) the opening handshake, visible in the trace: each side advertises
        #     {system, dialect, catalogue}. Surface only; semantics stay private.
        transcript: list[dict] = [
            {"from": "A", "advertisement": {"system": a.system, "dialect": a.dialect,
                                            "catalogue": cat_a}},
            {"from": "B", "advertisement": {"system": b.system, "dialect": b.dialect,
                                            "catalogue": cat_b}},
        ]

        proposed_by: dict[frozenset, set] = {}
        proposed_conf: dict[frozenset, float] = {}
        accepts: set[frozenset] = set()
        rejects: set[frozenset] = set()
        oracle_confirmed: set[frozenset] = set()
        oracle_refuted: set[frozenset] = set()
        a_ids, b_ids = set(a.by_id), set(b.by_id)
        total = {"total_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.0}
        turns = 0

        def record(side, turn: Turn):
            for p in turn.proposals:
                if side == "A":
                    pair = frozenset((p.my_id, p.their_id)) if p.my_id in a_ids and p.their_id in b_ids else None
                else:
                    pair = frozenset((p.their_id, p.my_id)) if p.my_id in b_ids and p.their_id in a_ids else None
                if pair:
                    proposed_by.setdefault(pair, set()).add(side)
                    proposed_conf[pair] = max(proposed_conf.get(pair, 0.0),
                                              float(getattr(p, "confidence", 0.0) or 0.0))
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

        self.last_transcript = transcript
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
            confidence={pair: (1.0 if pair in oracle_confirmed else proposed_conf.get(pair, 0.5))
                        for pair in confirmed},
        )
