"""Offline test of the construct-then-bind protocol (Track A / A1).

Injects a fake OpenAI client so no network is touched: step 1 returns a small
constructed reference, step 2 returns the gold correspondences. The test asserts the
pipeline wires end to end (construct -> bind -> reconciliation -> score) and produces a
correctly scored row.
"""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reconcile import Case                                                    # noqa: E402
from reconcile.metrics import score                                          # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack                   # noqa: E402
from reconcile.construct import (construct_reference, bind_through_reference,  # noqa: E402
                                 _ConstructedReference, _Entry, _BindResult, _Corr)

CASE_DIR = ROOT / "benchmark" / "cases" / "config_cross_domain"


def _completion(parsed):
    usage = types.SimpleNamespace(total_tokens=100, completion_tokens=60, prompt_tokens=40,
                                  completion_tokens_details=types.SimpleNamespace(reasoning_tokens=12))
    msg = types.SimpleNamespace(parsed=parsed)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


class _FakeClient:
    """Returns a constructed reference for step 1 and the gold pairs for step 2."""

    def __init__(self, gold_pairs):
        self._pairs = gold_pairs
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(parse=self._parse))

    def _parse(self, model, messages, response_format):
        if response_format is _ConstructedReference:
            entries = [_Entry(id=f"ref-{i}", label=f"entry{i}", definition="d", example="x")
                       for i in range(len(self._pairs))]
            return _completion(_ConstructedReference(entries=entries))
        if response_format is _BindResult:
            corrs = [_Corr(a_id=p["a"], b_id=p["b"], confidence=1.0) for p in self._pairs]
            return _completion(_BindResult(correspondences=corrs))
        raise AssertionError(f"unexpected response_format {response_format}")


def test_construct_then_bind_wires_and_scores():
    case = Case.load(CASE_DIR)
    gold_pairs = json.loads((CASE_DIR / "gold.json").read_text())["correspondences"]
    fake = _FakeClient(gold_pairs)

    cref, e1 = construct_reference(case.model_a, case.model_b, "fake", client=fake)
    assert len(cref.entries) == len(gold_pairs)
    assert e1["entries"] == len(gold_pairs)

    bres, e2 = bind_through_reference(case.model_a, case.model_b, cref, "fake",
                                      placement="both_cognitive", client=fake)
    assert len(bres.correspondences) == len(gold_pairs)

    stack = OpenAIAgentStack(use_reference=True, model="fake", client=fake)
    rec = stack.to_reconciliation(bres, case.model_a, case.model_b, {**e1, **e2}, "both_cognitive")
    s = score(rec, case.gold)

    # step 2 returned exactly the gold pairs, so the scored row is perfect
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["surviving_false_cognates"] == 0
