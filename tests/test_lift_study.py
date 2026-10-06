"""Offline checks for the agent lift (src/reconcile/lift.py): no API key, a stub client returns a
scripted lift. Verifies the lift produces the explanation layer and preserves everything else, that the
lift input withholds the fixture's own gloss/example (information hygiene - the agent lifts from the
schema surface, not from the answer), coverage accounting, and the fidelity proxy.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reconcile.model import Concept, SemanticModel                       # noqa: E402
from reconcile.lift import lift_model, gloss_fidelity, _LiftResult, _Lifted  # noqa: E402


# ---- a stub OpenAI-shaped client that returns a queued parsed _LiftResult ----------------
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


def _model():
    return SemanticModel(system="T", dialect="tapi", modules=(), concepts=[
        Concept(id="t.node", label="node", kind="node", gloss="FIXTURE-GLOSS-NODE",
                example="FIXTURE-EX-NODE", synonyms=("forwarding domain",), ref="forwarding-node",
                relations=({"rel": "part-of", "target": "t.topo"},), instances=("R1", "R2")),
        Concept(id="t.link", label="link", kind="link", gloss="FIXTURE-GLOSS-LINK",
                example="FIXTURE-EX-LINK", synonyms=(), ref="topo-link",
                relations=({"rel": "between", "target": "t.node"},), instances=("link-R1-R2",)),
    ])


def _lift_result(pairs):
    return _LiftResult(concepts=[_Lifted(id=i, gloss=g, example=e) for i, g, e in pairs])


def test_lift_produces_explanation_and_preserves_structure():
    sm = _model()
    parsed = _lift_result([("t.node", "a produced gloss for node", "produced example R9"),
                           ("t.link", "a produced gloss for link", "produced example link-R9")])
    client = StubClient([(parsed, 1234, 456)])
    lifted, eff = lift_model(sm, "gpt-test", client=client)

    by = {c.id: c for c in lifted.concepts}
    # explanation layer replaced with the agent's
    assert by["t.node"].gloss == "a produced gloss for node"
    assert by["t.node"].example == "produced example R9"
    # everything else preserved from the fixture
    assert by["t.node"].kind == "node"
    assert by["t.node"].synonyms == ("forwarding domain",)
    assert by["t.node"].ref == "forwarding-node"
    assert by["t.node"].relations == ({"rel": "part-of", "target": "t.topo"},)
    assert by["t.node"].instances == ("R1", "R2")
    assert eff["lift_coverage"] == 1.0
    assert eff["lift_reasoning_tokens"] == 456


def test_lift_input_hides_fixture_gloss_and_ref():
    """Information hygiene: the agent lifts from the schema surface. The fixture gloss, example, and
    reference binding must NOT appear in the prompt the lift sends."""
    sm = _model()
    parsed = _lift_result([("t.node", "g", "e"), ("t.link", "g", "e")])
    client = StubClient([(parsed, 1, 1)])
    lift_model(sm, "gpt-test", client=client)

    sent = json.dumps(client.calls[0])
    assert "FIXTURE-GLOSS-NODE" not in sent and "FIXTURE-EX-NODE" not in sent
    assert "FIXTURE-GLOSS-LINK" not in sent and "FIXTURE-EX-LINK" not in sent
    assert "forwarding-node" not in sent and "topo-link" not in sent   # ref binding withheld
    # but the schema surface IS present
    assert "forwarding domain" in sent and "t.node" in sent and "R1" in sent


def test_lift_coverage_counts_missing_concepts():
    sm = _model()
    parsed = _lift_result([("t.node", "only node lifted", "ex")])  # t.link omitted by the agent
    client = StubClient([(parsed, 1, 1)])
    lifted, eff = lift_model(sm, "gpt-test", client=client)
    by = {c.id: c for c in lifted.concepts}
    assert by["t.link"].gloss == "" and by["t.link"].example == ""    # honest miss
    assert eff["lift_coverage"] == 0.5


def test_to_dict_round_trips_and_matches_fixture_shape(tmp_path=None):
    """The agent-lift dump serialises a SemanticModel back to the fixture's JSON shape, round-trips
    through from_json, and carries the agent's gloss/example while preserving every other field."""
    sm = _model()
    parsed = _lift_result([("t.node", "produced gloss node", "produced ex node"),
                           ("t.link", "produced gloss link", "produced ex link")])
    lifted, _ = lift_model(sm, "gpt-test", client=StubClient([(parsed, 1, 1)]))

    d = lifted.to_dict(note="agent lift")
    assert list(d["concepts"][0].keys()) == [
        "id", "label", "synonyms", "kind", "gloss", "example", "ref", "relations", "instances"]
    assert d["note"] == "agent lift"

    # write and read back through the real loader
    p = Path(__file__).resolve().parent / "_tmp_agent_lift.json"
    try:
        p.write_text(json.dumps(d))
        back = SemanticModel.from_json(p)
        by = {c.id: c for c in back.concepts}
        assert by["t.node"].gloss == "produced gloss node"          # agent's explanation layer
        assert by["t.node"].ref == "forwarding-node"                 # source field preserved
        assert by["t.node"].relations == ({"rel": "part-of", "target": "t.topo"},)
        assert by["t.node"].instances == ("R1", "R2")
    finally:
        p.unlink(missing_ok=True)


def test_trace_records_evidence_and_asks_for_it():
    """--trace mode: the lift asks for a per-concept EVIDENCE note and returns it in effort['trace'],
    without disturbing the lifted model itself."""
    from reconcile.lift import _LiftResultTraced, _LiftedTraced
    sm = _model()
    parsed = _LiftResultTraced(concepts=[
        _LiftedTraced(id="t.node", evidence="kind=node; relation part-of->t.topo", gloss="g", example="e"),
        _LiftedTraced(id="t.link", evidence="kind=link; between->t.node", gloss="g", example="e")])
    client = StubClient([(parsed, 1, 1)])
    lifted, eff = lift_model(sm, "gpt-test", client=client, trace=True)
    assert "trace" in eff and eff["trace"]["t.node"].startswith("kind=node")
    assert "EVIDENCE" in json.dumps(client.calls[0])           # the prompt asked for it
    by = {c.id: c for c in lifted.concepts}                     # model unaffected
    assert by["t.node"].gloss == "g" and by["t.node"].ref == "forwarding-node"


def test_gloss_fidelity_range():
    assert gloss_fidelity("the same content words here", "same content words here") == 1.0
    assert gloss_fidelity("alpha beta gamma", "delta epsilon zeta") == 0.0
    assert gloss_fidelity("", "") == 1.0
    assert gloss_fidelity("something", "") == 0.0
    mid = gloss_fidelity("node forwarding domain topology", "node topology only")
    assert 0.0 < mid < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all lift-study offline checks passed")
