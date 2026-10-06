"""Reference-blind baseline: a classical label-and-structure matcher.

This is the control. It ignores the authors' reference bindings and proposes
candidate correspondences from surface tokens plus a small same-kind bonus, the
"label pass" of the paper. It does no gloss-and-example disambiguation, so a
lexically tempting false cognate is proposed and survives as a silent error. It
is what reconciliation looks like with neither a shared reference nor deep
cognition to fall back on.
"""
from __future__ import annotations

from reconcile.model import SemanticModel
from reconcile.reference import Reference
from reconcile.stacks.base import ReasoningStack, Reconciliation


def _jaccard(x: set[str], y: set[str]) -> float:
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


class BaselineMatcher(ReasoningStack):
    name = "baseline-label-matcher"
    uses_reference = False

    def __init__(self, threshold: float = 0.20, kind_bonus: float = 0.05):
        self.threshold = threshold
        self.kind_bonus = kind_bonus

    def score(self, ca, cb) -> float:
        s = _jaccard(ca.surface_tokens, cb.surface_tokens)
        if ca.kind and ca.kind == cb.kind:
            s += self.kind_bonus
        return s

    def reconcile(
        self,
        a: SemanticModel,
        b: SemanticModel,
        reference: Reference | None = None,
        placement: str = "both_cognitive",
    ) -> Reconciliation:
        proposed: list[frozenset] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()
        # Label pass: propose every cross pair above threshold (names alone).
        for ca in a.concepts:
            for cb in b.concepts:
                if self.score(ca, cb) >= self.threshold:
                    proposed.append(frozenset((ca.id, cb.id)))
                    matched_a.add(ca.id)
                    matched_b.add(cb.id)
        residual_a = [c.id for c in a.concepts if c.id not in matched_a]
        residual_b = [c.id for c in b.concepts if c.id not in matched_b]
        # Each proposed candidate would need a bilateral gloss/example check to
        # confirm or reject; the baseline does not perform it.
        work = {"candidates": len(proposed), "bilateral_checks": len(proposed), "binding_ops": 0}
        return Reconciliation(
            stack=self.name,
            uses_reference=False,
            placement=placement,
            proposed=proposed,
            residual_a=residual_a,
            residual_b=residual_b,
            work=work,
        )
