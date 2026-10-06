"""Sanity tests for the walking skeleton. Run: python -m pytest -q  (or python tests/test_harness.py)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconcile import BaselineMatcher, Case, ReferenceReconciler, run_case  # noqa: E402
from reconcile.model import Gold, SemanticModel, Concept  # noqa: E402
from reconcile.stacks.base import Reconciliation  # noqa: E402
from reconcile.metrics import score  # noqa: E402

CASE_DIR = ROOT / "benchmark" / "cases" / "config_tapi_teas"


def _case() -> Case:
    return Case.load(CASE_DIR)


def test_gold_self_scores_perfect():
    """A reconciliation equal to the gold correspondences scores precision=recall=1."""
    case = _case()
    rec = Reconciliation(
        stack="oracle", uses_reference=True, placement="both_cognitive",
        proposed=list(case.gold.correct_pairs),
    )
    m = score(rec, case.gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["surviving_false_cognates"] == 0


def test_baseline_surfaces_false_cognate():
    """The reference-blind label matcher proposes the planted false cognate."""
    case = _case()
    m = run_case(case, BaselineMatcher())
    assert m["surviving_false_cognates"] == 1, "baseline should let the false cognate through"
    assert m["recall"] < 1.0, "label-only matching should miss non-lexical correspondences"


def test_reference_preempts_false_cognate_and_is_complete():
    """The reference-aware reconciler pre-empts the false cognate and recovers all correspondences."""
    case = _case()
    m = run_case(case, ReferenceReconciler())
    assert m["surviving_false_cognates"] == 0, "reference should pre-empt the false cognate"
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["bilateral_checks"] == 0, "correspondences arrive as near-identity, no bilateral checks"


def test_reference_beats_baseline():
    case = _case()
    base = run_case(case, BaselineMatcher())
    ref = run_case(case, ReferenceReconciler())
    assert ref["precision"] >= base["precision"]
    assert ref["recall"] >= base["recall"]
    assert ref["surviving_false_cognates"] <= base["surviving_false_cognates"]


def test_residual_is_the_two_native_gaps():
    """Exactly the client-service-access (A) and trail-termination (B) concepts remain."""
    case = _case()
    ref = ReferenceReconciler().reconcile(case.model_a, case.model_b, reference=case.reference)
    assert set(ref.residual_a) == {"t.sip"}
    assert set(ref.residual_b) == {"i.ttp"}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
