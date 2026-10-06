#!/usr/bin/env python3
"""Offline test of the instance tool-use loop, driven by a scripted fake client (no API).

Exercises the full loop: interrogate the symmetric nodes, a refuting virtual_provision on
the svc-100 name trap, a confirming provision on the true ODU2 service, then submit. Asserts
the oracle counters, confirmed/refuted tracking, transcript, and scoring all behave.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.instance import load_instance_case                    # noqa: E402
from reconcile.stacks.agent_instance import InstanceAgentStack       # noqa: E402


def _usage(p=100, c=50, r=20):
    return SimpleNamespace(input_tokens=p, output_tokens=c, total_tokens=p + c,
                           output_tokens_details=SimpleNamespace(reasoning_tokens=r))


def _tool_call(cid, name, args_json):
    # a Responses-API function_call output item
    return {"type": "function_call", "call_id": cid, "name": name, "arguments": args_json}


def _completion(tool_calls=None, content=None):
    output = list(tool_calls or [])
    if content:
        output.append({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": content}]})
    return SimpleNamespace(output=output, usage=_usage())


class FakeClient:
    """Replays a scripted list of Responses-API responses, one per create() call."""
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        resp = self._script[self.calls]
        self.calls += 1
        return resp


def score(rec, case):
    proposed, correct = set(rec.proposed), case.correct_pairs
    fc = case.false_cognate_pairs
    tp = len(proposed & correct)
    fp = len(proposed - correct)
    fn = len(correct - proposed)
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "surviving_instance_fc": len(proposed & fc),
    }


def main():
    case = load_instance_case(ROOT / "benchmark" / "cases" / "instance_hard")
    correct = case.gold["correspondences"]

    # turn 1: interrogate the symmetric nodes; refute svc-100; confirm the ODU2 service
    turn1 = _completion(tool_calls=[
        _tool_call("c1", "interrogate", '{"individual_id":"a.r2","attribute":"serial"}'),
        _tool_call("c2", "interrogate", '{"individual_id":"b.nodeB","attribute":"serial"}'),
        _tool_call("c3", "virtual_provision", '{"a_id":"a.svc100","b_id":"b.svc100"}'),
        _tool_call("c4", "virtual_provision", '{"a_id":"a.cs_odu2","b_id":"b.tunnel1"}'),
    ])
    # turn 2: submit the correct alignment (a perfect agent, for plumbing/scoring)
    corr = [{"a_id": c["a"], "b_id": c["b"], "confidence": 0.9} for c in correct]
    submit_args = ('{"correspondences":' + str(corr).replace("'", '"') +
                   ',"residual_a":["a.svc100"],"residual_b":["b.svc100","b.r1"]}')
    turn2 = _completion(tool_calls=[_tool_call("c5", "submit_alignment", submit_args)])

    stack = InstanceAgentStack(case, model="fake", reference_variant="none",
                               budget=30, client=FakeClient([turn1, turn2]))
    rec = stack.reconcile(placement="both_cognitive")

    s = score(rec, case)
    checks = {
        "submitted": rec.submitted is True,
        "proposed==11": len(rec.proposed) == 11,
        "precision==1": s["precision"] == 1.0,
        "recall==1": s["recall"] == 1.0,
        "no surviving instance FC": s["surviving_instance_fc"] == 0,
        "interrogate count==2": rec.oracle_calls["interrogate"] == 2,
        "provision count==2": rec.oracle_calls["provision"] == 2,
        "ODU2 pair confirmed": frozenset(("a.cs_odu2", "b.tunnel1")) in rec.confirmed_pairs,
        "svc-100 NOT confirmed": frozenset(("a.svc100", "b.svc100")) not in rec.confirmed_pairs,
        "residual_a has svc100": "a.svc100" in rec.residual_a,
        "transcript has provision refute":
            any(t.get("step") == "virtual_provision" and t["result"]["answer"].get("confirmed") is False
                for t in rec.transcript),
        "effort turns==2": rec.effort["turns"] == 2,
        "reasoning tokens summed": rec.effort["reasoning_tokens"] == 40,
    }
    # both_inert: oracle tools withheld; a static submit still scores
    stack_bi = InstanceAgentStack(case, model="fake", reference_variant="none",
                                  budget=30, client=FakeClient([turn2]))
    rec_bi = stack_bi.reconcile(placement="both_inert")
    checks["both_inert submits, no oracle calls"] = (
        rec_bi.submitted and rec_bi.oracle_calls == {"interrogate": 0, "provision": 0})

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nSCORE:", s, "| oracle:", rec.oracle_calls,
          "| confirmed:", [set(p) for p in rec.confirmed_pairs])
    print("\nTRANSCRIPT (both_cognitive):")
    for t in rec.transcript:
        print("  ", t.get("step"), {k: v for k, v in t.items() if k != "step"})
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
