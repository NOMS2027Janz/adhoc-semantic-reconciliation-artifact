"""Instrumented two-agent stack: adds a first-person PLANNING VOICE to every turn.

Non-destructive. It subclasses ``TwoAgentStack`` and overrides only the per-turn model
call, so the base stack, its schema, and every existing result are untouched. Each turn the
agent now also fills a ``plan`` object -- its own account of where the reconciliation stands,
what it most needs next and why, its strategy for the turn, how close it judges itself to a
confident close, what it would do next, and whether to continue / stop / escalate. The field
flows into the saved transcript automatically, because the loop records ``turn.model_dump()``.

Used for the orchestration paper: the ``plan`` channel is the agent's orchestration
meta-reasoning, made legible for annotated, commented traces.
"""
from __future__ import annotations

import json
import sys
import time

from pydantic import BaseModel

from reconcile.stacks.agent_twoagent import (
    TwoAgentStack,
    Turn,
    _system,
    _effort_from_usage,
)


class Plan(BaseModel):
    """The agent's first-person orchestration reasoning for one turn."""

    situation: str   # my read of where the reconciliation stands now (settled vs open)
    need: str        # the ONE thing I most need to learn or settle next, and WHY it is the priority
    strategy: str    # my plan for THIS turn: what I will ask, propose, test, or ratify, and why
    closeness: float # my own estimate in [0,1] of how close I am to a confident, verified close
    next_step: str   # what I would do next if I am not yet done
    decision: str    # exactly one of "continue", "stop", "escalate"


class TurnI(Turn):
    """A turn, extended with the planning channel."""

    plan: Plan


_PLAN_INSTR = (
    "\n\nORCHESTRATION PLAN (required every turn). Before you act, fill the 'plan' object with "
    "your own first-person reasoning about the investigation, grounded in the transcript so far:\n"
    "  - situation: your read of where the reconciliation stands now -- what is settled, what is "
    "still open.\n"
    "  - need: the ONE thing you most need to learn or settle next, and why it is the priority now.\n"
    "  - strategy: your plan for THIS turn -- which of the other's concepts to ask about, which "
    "correspondences to propose, which to test with a decisive experiment if one is available, which "
    "outstanding proposals to ratify, and in what order, each with its reason.\n"
    "  - closeness: your own estimate in [0,1] of how close you are to a confident, verified close.\n"
    "  - next_step: what you would do next if you are not yet done.\n"
    "  - decision: exactly one of 'continue' (work remains), 'stop' (you judge the close complete), "
    "or 'escalate' (something here is beyond what you can settle and needs a person or a tool you "
    "lack).\n"
    "The plan must reflect the ACTUAL state and your real reasons, not a generic recipe, and your "
    "actions this turn must be consistent with it."
)


class TwoAgentStackInstrumented(TwoAgentStack):
    """``TwoAgentStack`` with a per-turn planning voice (the ``plan`` channel)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.name + "+plan"

    def _turn(self, client, side, other, private, their_catalogue, transcript):
        system = _system(
            side, other, self.uses_reference,
            has_oracle=self.oracle is not None,
            has_frame=self.shared_frame is not None,
        ) + _PLAN_INSTR
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
            response_format=TurnI,
        )
        elapsed = time.time() - t0
        turn = completion.choices[0].message.parsed
        eff = _effort_from_usage(getattr(completion, "usage", None))
        eff["latency_s"] = round(elapsed, 2)
        # live progress (per study convention): one line per completed agent turn
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
