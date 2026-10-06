"""A stronger classical ontology matcher (reference-blind): the prior-art bar.

Representative of the pre-LLM ontology-matching family (the lexical-plus-structural
approach LogMap and AgreementMakerLight use for candidate generation): it matches on
labels and synonyms, on the tokens of the disambiguating definition and example, and on
shallow structure (kind agreement and relation-neighbourhood overlap), and it emits a
1:1 alignment by greedy best-match above a threshold. It uses everything a descriptor
method can - names, gloss, and structure - but has no semantic understanding, so it
cannot tell a true correspondence from a lexically and structurally tempting false
cognate. It is the strongest form of the ceiling the study says only cognition passes,
and a fairer prior-art baseline than the plain label matcher.
"""
from __future__ import annotations

import re

from reconcile.model import SemanticModel
from reconcile.reference import Reference
from reconcile.stacks.base import ReasoningStack, Reconciliation

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(x: set[str], y: set[str]) -> float:
    return len(x & y) / len(x | y) if x and y else 0.0


def _rel_types(c) -> set[str]:
    return {r.get("rel", "") for r in c.relations}


class ClassicalMatcher(ReasoningStack):
    """Lexical + definitional + structural matcher with a 1:1 greedy alignment."""

    name = "classical (lexical+structural)"
    uses_reference = False

    def __init__(self, threshold: float = 0.30,
                 w_label: float = 0.55, w_def: float = 0.30, w_struct: float = 0.15):
        self.threshold = threshold
        self.w_label = w_label
        self.w_def = w_def
        self.w_struct = w_struct

    def score(self, ca, cb) -> float:
        label = _jaccard(ca.surface_tokens, cb.surface_tokens)
        defn = _jaccard(_tokens(f"{ca.gloss} {ca.example}"), _tokens(f"{cb.gloss} {cb.example}"))
        kind = 1.0 if (ca.kind and ca.kind == cb.kind) else 0.0
        rel = _jaccard(_rel_types(ca), _rel_types(cb))
        struct = 0.5 * kind + 0.5 * rel
        return self.w_label * label + self.w_def * defn + self.w_struct * struct

    def reconcile(self, a: SemanticModel, b: SemanticModel,
                  reference: Reference | None = None,
                  placement: str = "both_cognitive") -> Reconciliation:
        candidates = []
        for ca in a.concepts:
            for cb in b.concepts:
                s = self.score(ca, cb)
                if s >= self.threshold:
                    candidates.append((s, ca.id, cb.id))
        candidates.sort(reverse=True)  # best first
        proposed: list[frozenset] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()
        for _s, aid, bid in candidates:  # greedy 1:1 assignment
            if aid in matched_a or bid in matched_b:
                continue
            proposed.append(frozenset((aid, bid)))
            matched_a.add(aid)
            matched_b.add(bid)
        residual_a = [c.id for c in a.concepts if c.id not in matched_a]
        residual_b = [c.id for c in b.concepts if c.id not in matched_b]
        work = {"candidates": len(candidates), "bilateral_checks": len(proposed), "binding_ops": 0}
        return Reconciliation(
            stack=self.name, uses_reference=False, placement=placement,
            proposed=proposed, residual_a=residual_a, residual_b=residual_b, work=work,
        )
