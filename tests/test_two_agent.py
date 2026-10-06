"""Offline checks for the two-agent negotiation stack: no API key, a stub client returns
scripted turns. Verifies information asymmetry (each agent's prompt carries only its own
model in full, the other only as a surface catalogue), the propose-then-ratify confirmation,
residual computation, and that effort is summed across turns.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reconcile.model import Concept, SemanticModel                       # noqa: E402
from reconcile.stacks.agent_twoagent import (                            # noqa: E402
    TwoAgentStack, Turn, Proposal, Ratification, Experiment)
from reconcile.schema_oracle import ExperimentResult                    # noqa: E402


# ---- a stub OpenAI-shaped client that returns queued (Turn, total_tokens, reasoning) -----
class _Details:
    def __init__(self, rt): self.reasoning_tokens = rt


class _Usage:
    def __init__(self, tt, rt): self.total_tokens = tt; self.completion_tokens_details = _Details(rt)


class _Msg:
    def __init__(self, parsed): self.parsed = parsed


class _Choice:
    def __init__(self, parsed): self.message = _Msg(parsed)


class _Completion:
    def __init__(self, parsed, tt, rt): self.choices = [_Choice(parsed)]; self.usage = _Usage(tt, rt)


class _Completions:
    def __init__(self, outer): self._outer = outer

    def parse(self, model, messages, response_format):
        self._outer.calls.append(messages)
        parsed, tt, rt = self._outer.queue.pop(0)
        return _Completion(parsed, tt, rt)


class _Chat:
    def __init__(self, outer): self.completions = _Completions(outer)


class StubClient:
    def __init__(self, queue): self.queue = list(queue); self.calls = []; self.chat = _Chat(self)


def _models():
    a = SemanticModel(system="A", dialect="da", modules=(), concepts=[
        Concept(id="a1", label="alpha", kind="service", gloss="A-PRIVATE-GLOSS-1", example="ex"),
        Concept(id="a2", label="ay-two", kind="node", gloss="A-PRIVATE-GLOSS-2", example="ex"),
    ])
    b = SemanticModel(system="B", dialect="db", modules=(), concepts=[
        Concept(id="b1", label="beta", kind="service", gloss="B-PRIVATE-GLOSS-1", example="ex"),
        Concept(id="b2", label="bee-two", kind="link", gloss="B-PRIVATE-GLOSS-2", example="ex"),
    ])
    return a, b


def test_negotiation_confirms_by_ratification_and_sums_effort():
    a, b = _models()
    turn_a = Turn(answers=[], questions=[],
                  proposals=[Proposal(my_id="a1", their_id="b1", confidence=0.9, rationale="same service")],
                  ratifications=[], no_counterpart=[], done=True)
    turn_b = Turn(answers=[], questions=[],
                  proposals=[],
                  ratifications=[Ratification(their_id="a1", my_id="b1", accept=True, reason="agree")],
                  no_counterpart=["b2"], done=True)
    client = StubClient([(turn_a, 100, 40), (turn_b, 120, 50)])
    stack = TwoAgentStack(use_reference=False, model="stub", client=client, max_rounds=1)
    rec = stack.reconcile(a, b, placement="both_cognitive")

    assert rec.proposed == [frozenset(("a1", "b1"))], rec.proposed
    assert rec.residual_a == ["a2"], rec.residual_a
    assert rec.residual_b == ["b2"], rec.residual_b
    assert rec.work["turns"] == 2
    assert rec.effort["total_tokens"] == 220
    assert rec.effort["reasoning_tokens"] == 90
    print("ok: confirm-by-ratification, residual, effort summed")


def test_information_asymmetry_in_prompts():
    a, b = _models()
    done = Turn(answers=[], questions=[], proposals=[], ratifications=[], no_counterpart=[], done=True)
    client = StubClient([(done, 10, 0), (done, 10, 0)])
    stack = TwoAgentStack(use_reference=False, model="stub", client=client, max_rounds=1)
    stack.reconcile(a, b, placement="both_cognitive")

    # first call is Agent A's turn: its user payload must carry model A in full (with gloss),
    # model B only as a surface catalogue (no gloss), and must not leak B's private gloss text.
    a_payload = json.loads(client.calls[0][1]["content"])
    assert {c["id"] for c in a_payload["your_model"]} == {"a1", "a2"}
    assert all("gloss" in c for c in a_payload["your_model"])
    assert {c["id"] for c in a_payload["agent_B_catalogue"]} == {"b1", "b2"}
    assert all("gloss" not in c for c in a_payload["agent_B_catalogue"])
    assert "B-PRIVATE-GLOSS-1" not in client.calls[0][1]["content"]
    assert "B-PRIVATE-GLOSS-2" not in client.calls[0][1]["content"]

    # second call is Agent B's turn: mirror-image asymmetry
    b_payload = json.loads(client.calls[1][1]["content"])
    assert {c["id"] for c in b_payload["your_model"]} == {"b1", "b2"}
    assert "A-PRIVATE-GLOSS-1" not in client.calls[1][1]["content"]
    print("ok: information asymmetry holds in both agents' prompts")


def test_rejection_blocks_confirmation():
    a, b = _models()
    turn_a = Turn(answers=[], questions=[],
                  proposals=[Proposal(my_id="a1", their_id="b1", confidence=0.6, rationale="maybe")],
                  ratifications=[], no_counterpart=[], done=True)
    turn_b = Turn(answers=[], questions=[], proposals=[],
                  ratifications=[Ratification(their_id="a1", my_id="b1", accept=False, reason="false cognate")],
                  no_counterpart=[], done=True)
    client = StubClient([(turn_a, 1, 0), (turn_b, 1, 0)])
    stack = TwoAgentStack(model="stub", client=client, max_rounds=1)
    rec = stack.reconcile(a, b, placement="both_cognitive")
    assert rec.proposed == [], rec.proposed        # owner rejected: not confirmed
    assert set(rec.residual_a) == {"a1", "a2"}
    print("ok: an owner's rejection blocks a one-sided proposal")


class StubOracle:
    """A decisive virtual network for the test: (a1,b1) is real, (a2,b2) is a false cognate."""
    def __init__(self):
        self.calls = 0
        self.verdicts = {frozenset(("a1", "b1")): True, frozenset(("a2", "b2")): False}

    def virtual_provision(self, a_id, b_id):
        self.calls += 1
        pair = frozenset((a_id, b_id))
        conf = self.verdicts.get(pair, False)
        rel = "identity" if conf else ("false-cognate" if pair in self.verdicts else "no-correspondence")
        a = a_id if a_id.startswith("a") else b_id
        b = b_id if b_id.startswith("b") else a_id
        return ExperimentResult(ok=True, confirmed=conf, relation=rel, a_id=a, b_id=b)


def test_experiment_confirms_and_refutes():
    a, b = _models()
    # A proposes a wrong pair (a2,b2) AND runs experiments on both (a1,b1) and (a2,b2)
    turn_a = Turn(answers=[], questions=[],
                  proposals=[Proposal(my_id="a2", their_id="b2", confidence=0.7, rationale="looks alike")],
                  ratifications=[],
                  experiments=[Experiment(my_id="a1", their_id="b1", question="same?"),
                               Experiment(my_id="a2", their_id="b2", question="false cognate?")],
                  no_counterpart=[], done=True)
    turn_b = Turn(answers=[], questions=[], proposals=[], ratifications=[], experiments=[],
                  no_counterpart=[], done=True)
    client = StubClient([(turn_a, 50, 10), (turn_b, 10, 0)])
    oracle = StubOracle()
    stack = TwoAgentStack(model="stub", client=client, max_rounds=1, oracle=oracle)
    rec = stack.reconcile(a, b, placement="both_cognitive")

    # the experiment CONFIRMS (a1,b1) even though it was never proposed or ratified,
    # and REFUTES (a2,b2) even though A proposed it.
    assert rec.proposed == [frozenset(("a1", "b1"))], rec.proposed
    assert rec.residual_a == ["a2"], rec.residual_a
    assert rec.residual_b == ["b2"], rec.residual_b
    assert rec.work["experiments"] == 2, rec.work
    print("ok: decisive experiment confirms a real pair and refutes a false cognate")


def test_shared_frame_reaches_both_agents():
    """The constructed shared frame (the candidate-surfacing act) must be injected into BOTH the
    system instruction and the user payload, and absence of a frame must change nothing."""
    from reconcile.reference import Reference, ReferenceEntry
    a, b = _models()
    done = Turn(answers=[], questions=[], proposals=[], ratifications=[], experiments=[],
                no_counterpart=[], done=True)
    frame = Reference(id="constructed", kind="constructed", entries=[
        ReferenceEntry(id="e1", label="thing", definition="d", example="ex", cls="c", synonyms=())])
    client = StubClient([(done, 10, 0), (done, 10, 0)])
    stack = TwoAgentStack(use_reference=False, model="stub", client=client, max_rounds=1,
                          shared_frame=frame)
    stack.reconcile(a, b, placement="both_cognitive")
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["shared_frame"][0]["id"] == "e1", payload.get("shared_frame")
    assert "SHARED REFERENCE FRAME" in client.calls[0][0]["content"]
    assert stack.name == "two-agent+frame/no-ref", stack.name

    # backward-compatible: no frame => no shared_frame key and no frame instruction
    c2 = StubClient([(done, 10, 0), (done, 10, 0)])
    TwoAgentStack(use_reference=False, model="stub", client=c2, max_rounds=1).reconcile(
        a, b, placement="both_cognitive")
    assert "shared_frame" not in json.loads(c2.calls[0][1]["content"])
    assert "SHARED REFERENCE FRAME" not in c2.calls[0][0]["content"]
    print("ok: shared frame reaches both agents; absence changes nothing")


if __name__ == "__main__":
    test_negotiation_confirms_by_ratification_and_sums_effort()
    test_information_asymmetry_in_prompts()
    test_rejection_blocks_confirmation()
    test_experiment_confirms_and_refutes()
    test_shared_frame_reaches_both_agents()
    print("\nall two-agent offline checks passed")
