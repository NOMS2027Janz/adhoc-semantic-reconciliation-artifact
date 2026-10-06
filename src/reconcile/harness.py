"""The evaluation harness: run a stack on a case and score it against gold."""
from __future__ import annotations

from reconcile.model import Case
from reconcile.metrics import score
from reconcile.stacks.base import ReasoningStack


def run_case(case: Case, stack: ReasoningStack, placement: str = "both_cognitive") -> dict:
    reference = case.reference if stack.uses_reference else None
    rec = stack.reconcile(case.model_a, case.model_b, reference=reference, placement=placement)
    row = {"case": case.name}
    row.update(score(rec, case.gold))
    return row
