"""A decisive virtual-experiment oracle for the schema (concept) settings.

The theory's completion argument for the fully-cognitive case rests on the *decisive virtual
experiment*: because reconciliation runs on the models, not the live network, two live agents
can provision a candidate correspondence in virtual space, operate it, read it back, and check
the invariants a correct translation must preserve. Any correspondence question is therefore
confirmed or refuted -- not deferred.

This oracle is that virtual network. It embodies the ground truth the two live systems jointly
determine (authored as the case's gold: the true correspondences, the planted false cognates,
and the invariants). It is NOT visible to the agents; they may only *provision* a candidate pair
through it and read the verdict back, exactly as they could operate their own models against each
other. `budget` caps the number of experiments so a run cannot brute-force every pair; the agents
must choose which candidates to test, as they would spend real effort.

A verdict is decisive:
  * confirmed / identity      -- provisioning through the pair preserves the invariants: same thing.
  * refuted / false-cognate   -- a shared surface word, but the invariants diverge: not the same.
  * refuted / no-correspondence -- no counterpart; the provision has nothing to bind to.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from reconcile.model import Case


@dataclass
class ExperimentResult:
    ok: bool                      # False only when the budget is exhausted (no live experiment left)
    confirmed: bool = False
    relation: str = ""            # "identity" | "false-cognate" | "no-correspondence" | "over-budget"
    a_id: str = ""
    b_id: str = ""


class SchemaOracle:
    """Deterministic provision-and-read-back over a schema case, grounded in its gold."""

    def __init__(self, case: Case, budget: int | None = None):
        self.correct = case.gold.correct_pairs          # set[frozenset({a,b})]
        self.false_cog = case.gold.false_cognate_pairs
        self.a_ids = set(case.model_a.by_id)
        self.b_ids = set(case.model_b.by_id)
        self.budget = budget
        self.calls = 0
        self.log: list[ExperimentResult] = []

    def virtual_provision(self, a_id: str, b_id: str) -> ExperimentResult:
        # normalise: a_id must name a model-A concept, b_id a model-B concept
        if a_id in self.b_ids and b_id in self.a_ids:
            a_id, b_id = b_id, a_id
        if a_id not in self.a_ids or b_id not in self.b_ids:
            r = ExperimentResult(ok=True, confirmed=False, relation="no-correspondence",
                                 a_id=a_id, b_id=b_id)
            self.log.append(r)
            return r
        if self.budget is not None and self.calls >= self.budget:
            return ExperimentResult(ok=False, relation="over-budget", a_id=a_id, b_id=b_id)
        self.calls += 1
        pair = frozenset((a_id, b_id))
        if pair in self.correct:
            rel, conf = "identity", True
        elif pair in self.false_cog:
            rel, conf = "false-cognate", False
        else:
            rel, conf = "no-correspondence", False
        r = ExperimentResult(ok=True, confirmed=conf, relation=rel, a_id=a_id, b_id=b_id)
        self.log.append(r)
        return r
