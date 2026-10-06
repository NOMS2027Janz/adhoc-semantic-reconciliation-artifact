#!/usr/bin/env python3
"""Offline test of the reach interrogation axis (no API).

Checks that the interrogation "reach" levels (none | lookup | discovery | full) each expose
exactly the right tools and enforce the right rules:
  - the tool list and the prompt advertise only the offered tools;
  - "lookup" refuses a bare interrogate and does NOT leak the available-attribute list
    (no discovery), while "discovery" allows a bare interrogate (all facts);
  - "none" runs a static submit with no oracle calls;
  - "full" is unchanged from the legacy repertoire (regression).
"""
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.instance import load_instance_case, Oracle              # noqa: E402
from reconcile.stacks.agent_instance import (                          # noqa: E402
    InstanceAgentStack, _tools_for, INTERROGATION_LEVELS)


def _usage(p=100, c=50, r=20):
    return SimpleNamespace(input_tokens=p, output_tokens=c, total_tokens=p + c,
                           output_tokens_details=SimpleNamespace(reasoning_tokens=r))


def _tool_call(cid, name, args_json):
    return {"type": "function_call", "call_id": cid, "name": name, "arguments": args_json}


def _completion(tool_calls=None, content=None):
    output = list(tool_calls or [])
    if content:
        output.append({"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": content}]})
    return SimpleNamespace(output=output, usage=_usage())


class FakeClient:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        resp = self._script[self.calls]
        self.calls += 1
        return resp


def _tool_names(interrogation):
    return {t["name"] for t in _tools_for(interrogation)}


def main():
    case = load_instance_case(ROOT / "benchmark" / "cases" / "instance_hard")
    checks = {}

    # --- tool sets per interrogation -------------------------------------------------------
    checks["none: submit only"] = _tool_names("none") == {"submit_alignment"}
    checks["lookup: interrogate+submit, no provision"] = _tool_names("lookup") == {
        "interrogate", "submit_alignment"}
    checks["discovery: interrogate+submit, no provision"] = _tool_names("discovery") == {
        "interrogate", "submit_alignment"}
    checks["full: interrogate+provision+submit"] = _tool_names("full") == {
        "interrogate", "virtual_provision", "submit_alignment"}

    # lookup's interrogate schema requires the attribute; discovery's does not
    look_iga = next(t for t in _tools_for("lookup") if t["name"] == "interrogate")
    disc_iga = next(t for t in _tools_for("discovery") if t["name"] == "interrogate")
    checks["lookup interrogate requires attribute"] = "attribute" in look_iga["parameters"]["required"]
    checks["discovery interrogate: attribute optional"] = "attribute" not in disc_iga["parameters"]["required"]

    # --- prompt advertises exactly the offered tools ------------------------------------
    def sysprompt(aff):
        return InstanceAgentStack(case, model="fake", interrogation=aff)._system("both_cognitive")
    p_none, p_look, p_disc, p_full = (sysprompt(a) for a in INTERROGATION_LEVELS)
    checks["none prompt: no tool offer"] = ("virtual_provision" not in p_none
                                            and "no tools to probe" in p_none)
    checks["lookup prompt: no provision, must name attribute"] = (
        "virtual_provision" not in p_look and "must name the attribute" in p_look)
    checks["discovery prompt: no provision, omit ok"] = (
        "virtual_provision" not in p_disc and "omit the attribute" in p_disc)
    checks["full prompt: provision offered"] = "virtual_provision" in p_full

    # --- dispatch-level gating of discovery ---------------------------------------------
    # pick a live individual that has interrogable facts, and a hit/miss attribute
    oracle_facts = case.traps.get("oracle", {}).get("interrogate", {})
    live_id = next(iter(oracle_facts))
    hit_attr = next(iter(oracle_facts[live_id]))

    stack_look = InstanceAgentStack(case, model="fake", interrogation="lookup")
    orc = Oracle(case, "both_cognitive", inert_side="b", budget=None)
    # bare interrogate is refused under lookup
    bare = stack_look._dispatch(orc, "interrogate", {"individual_id": live_id})
    checks["lookup: bare interrogate refused"] = (bare["ok"] is False
                                                  and "requires an attribute" in bare["message"])
    # a named MISS does not leak the available-attribute list under lookup
    miss = stack_look._dispatch(orc, "interrogate",
                                {"individual_id": live_id, "attribute": "no_such_attr_xyz"})
    checks["lookup: no attribute-list leak on miss"] = "interrogable attributes:" not in miss["message"]
    # a named HIT still returns the fact
    hit = stack_look._dispatch(orc, "interrogate",
                               {"individual_id": live_id, "attribute": hit_attr})
    checks["lookup: named hit returns fact"] = hit["ok"] is True and hit_attr in (hit["answer"] or {})

    stack_disc = InstanceAgentStack(case, model="fake", interrogation="discovery")
    orc2 = Oracle(case, "both_cognitive", inert_side="b", budget=None)
    disc = stack_disc._dispatch(orc2, "interrogate", {"individual_id": live_id})
    checks["discovery: bare interrogate returns facts"] = (disc["ok"] is True
                                                           and bool(disc["answer"]))

    # --- functional loop: 'none' submits with zero oracle calls -------------------------
    correct = case.gold["correspondences"]
    corr = [{"a_id": c["a"], "b_id": c["b"], "confidence": 0.9} for c in correct]
    submit_args = ('{"correspondences":' + str(corr).replace("'", '"') +
                   ',"residual_a":["a.svc100"],"residual_b":["b.svc100","b.r1"]}')
    turn_submit = _completion(tool_calls=[_tool_call("c1", "submit_alignment", submit_args)])
    stack_none = InstanceAgentStack(case, model="fake", interrogation="none",
                                    client=FakeClient([turn_submit]))
    rec_none = stack_none.reconcile(placement="both_cognitive")
    checks["none: submits, zero oracle calls"] = (
        rec_none.submitted and rec_none.oracle_calls == {"interrogate": 0, "provision": 0})
    checks["none: interrogation recorded in transcript"] = any(
        t.get("step") == "prompt" and t.get("interrogation") == "none" for t in rec_none.transcript)

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
