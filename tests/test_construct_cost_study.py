"""Offline test of the construct-then-bind COST-BENEFIT runner (Track A / A6).

Injects a fake OpenAI client (no network). Verifies that the constructed and baseline
row-builders wire end to end and, in particular, that the cognition-spend columns are
recorded correctly: the constructed condition sums the construct step and the bind step,
while a baseline condition records only the bind spend with construct spend zero.
"""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile import Case                                                     # noqa: E402
from reconcile.construct import (construct_reference, bind_through_reference,   # noqa: E402
                                 _ConstructedReference, _Entry, _BindResult, _Corr)
from reconcile.stacks.agent_openai import OpenAIAgentStack                     # noqa: E402
import construct_cost_study as ccs                                            # noqa: E402

CASE_DIR = ROOT / "benchmark" / "cases" / "config_cross_domain"


def _completion(parsed, reasoning):
    usage = types.SimpleNamespace(total_tokens=reasoning * 5, completion_tokens=reasoning * 3,
                                  prompt_tokens=reasoning * 2,
                                  completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning))
    msg = types.SimpleNamespace(parsed=parsed)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


class _FakeClient:
    def __init__(self, gold_pairs):
        self._pairs = gold_pairs
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(parse=self._parse))

    def _parse(self, model, messages, response_format):
        if response_format is _ConstructedReference:
            entries = [_Entry(id=f"ref-{i}", label=f"e{i}", definition="d", example="x")
                       for i in range(len(self._pairs))]
            return _completion(_ConstructedReference(entries=entries), reasoning=12)
        if response_format is _BindResult:
            corrs = [_Corr(a_id=p["a"], b_id=p["b"], confidence=1.0) for p in self._pairs]
            return _completion(_BindResult(correspondences=corrs), reasoning=12)
        raise AssertionError(f"unexpected response_format {response_format}")


def _setup():
    case = Case.load(CASE_DIR)
    gold_pairs = json.loads((CASE_DIR / "gold.json").read_text())["correspondences"]
    fake = _FakeClient(gold_pairs)
    cref, e1 = construct_reference(case.model_a, case.model_b, "fake", client=fake)
    bres, e2 = bind_through_reference(case.model_a, case.model_b, cref, "fake",
                                      placement="both_cognitive", client=fake)
    return case, gold_pairs, e1, e2, bres


def test_constructed_row_sums_construct_and_bind_spend():
    case, gold_pairs, e1, e2, bres = _setup()
    stack = OpenAIAgentStack(use_reference=True, model="fake")
    rec = stack.to_reconciliation(bres, case.model_a, case.model_b, {**e1, **e2}, "both_cognitive")
    row = ccs._constructed_row(case, "fake", "both_cognitive", 0, rec, e1, e2)

    assert set(row.keys()) == set(ccs.COLS)
    assert row["precision"] == 1.0 and row["recall"] == 1.0
    assert row["surviving_false_cognates"] == 0
    assert row["constructed_entries"] == len(gold_pairs)
    # construct (12) + bind (12) reasoning tokens are summed
    assert row["construct_reasoning_tokens"] == 12
    assert row["bind_reasoning_tokens"] == 12
    assert row["total_reasoning_tokens"] == 24


def test_base_row_records_bind_spend_only():
    case, gold_pairs, e1, e2, bres = _setup()
    stack = OpenAIAgentStack(use_reference=False, model="fake")
    # a baseline binding: reuse the same correspondences, with a bind-only effort record
    rec = stack.to_reconciliation(bres, case.model_a, case.model_b,
                                  {"reasoning_tokens": 7, "total_tokens": 30}, "both_cognitive")
    row = ccs._base_row(case, "no-ref", "fake", "both_cognitive", 0, rec)

    assert set(row.keys()) == set(ccs.COLS)
    assert row["construct_reasoning_tokens"] == 0
    assert row["bind_reasoning_tokens"] == 7
    assert row["total_reasoning_tokens"] == 7
    assert row["total_tokens"] == 30
    assert row["precision"] == 1.0 and row["recall"] == 1.0


def test_conditions_and_columns_stable():
    assert ccs.CONDITIONS == ("no-ref", "constructed", "given-ref")
    for key in ("construct_reasoning_tokens", "bind_reasoning_tokens", "total_reasoning_tokens"):
        assert key in ccs.COLS
