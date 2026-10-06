"""Reference-aware reconciler: bind once, derive correspondences.

This is the treatment. Each side has bound its concepts to the thin shared
reference. Correspondence is then near-identity: two concepts correspond exactly
when they bind to the same reference entry. A false cognate is pre-empted,
because two terms that merely share a label bind to different entries. Concepts
that bind to an entry the other side does not reach are the residual, closed by
self-extension. No bilateral checking is needed; each binding is made once and
reused.

It is deliberately shallow, like the baseline, so the contrast isolates the
effect of the reference alone. Deep cognition (an LM agent) is a separate stack.
"""
from __future__ import annotations

from collections import defaultdict

from reconcile.model import SemanticModel
from reconcile.reference import Reference
from reconcile.stacks.base import ReasoningStack, Reconciliation


class ReferenceReconciler(ReasoningStack):
    name = "reference-reconciler"
    uses_reference = True

    def reconcile(
        self,
        a: SemanticModel,
        b: SemanticModel,
        reference: Reference | None = None,
        placement: str = "both_cognitive",
    ) -> Reconciliation:
        by_ref_a: dict[str, list[str]] = defaultdict(list)
        by_ref_b: dict[str, list[str]] = defaultdict(list)
        for c in a.concepts:
            if c.ref:
                by_ref_a[c.ref].append(c.id)
        for c in b.concepts:
            if c.ref:
                by_ref_b[c.ref].append(c.id)

        proposed: list[frozenset] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()
        shared = set(by_ref_a) & set(by_ref_b)
        for ref_id in shared:
            for aid in by_ref_a[ref_id]:
                for bid in by_ref_b[ref_id]:
                    proposed.append(frozenset((aid, bid)))
                    matched_a.add(aid)
                    matched_b.add(bid)

        residual_a = [c.id for c in a.concepts if c.id not in matched_a]
        residual_b = [c.id for c in b.concepts if c.id not in matched_b]
        # Correspondences arrive as near-identity, auto-confirmed; no bilateral
        # checks. Work is one binding per concept, made once and reused.
        binding_ops = sum(1 for c in a.concepts if c.ref) + sum(1 for c in b.concepts if c.ref)
        work = {"candidates": len(proposed), "bilateral_checks": 0, "binding_ops": binding_ops}
        return Reconciliation(
            stack=self.name,
            uses_reference=True,
            placement=placement,
            proposed=proposed,
            residual_a=residual_a,
            residual_b=residual_b,
            work=work,
        )
