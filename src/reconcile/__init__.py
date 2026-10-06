"""Ad hoc semantic reconciliation: a study harness.

Public surface for the walking skeleton.
"""
from reconcile.model import Case, Concept, Gold, SemanticModel
from reconcile.reference import Reference, ReferenceEntry
from reconcile.harness import run_case
from reconcile.metrics import score, METRIC_COLUMNS
from reconcile.stacks.base import ReasoningStack, Reconciliation
from reconcile.stacks.baseline_matcher import BaselineMatcher
from reconcile.stacks.reference_matcher import ReferenceReconciler

__all__ = [
    "Case", "Concept", "Gold", "SemanticModel",
    "Reference", "ReferenceEntry",
    "run_case", "score", "METRIC_COLUMNS",
    "ReasoningStack", "Reconciliation",
    "BaselineMatcher", "ReferenceReconciler",
]
