#!/usr/bin/env python3
"""Offline test of the intent tool-use loop, driven by a scripted fake client (no API).

Exercises all three phases with a 'perfect agent' script and asserts: the oracle counters,
the placement gating (an inert provider refuses live feasibility; both-inert refuses the
policy), the transcript, and that the submitted answers match the derived gold.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.intent import load_intent_case                          # noqa: E402
from reconcile.stacks.agent_intent import IntentAgentStack            # noqa: E402


def _usage(p=100, c=50, r=20):
    return SimpleNamespace(input_tokens=p, output_tokens=c, total_tokens=p + c,
                           output_tokens_details=SimpleNamespace(reasoning_tokens=r))


def _call(cid, name, args_json):
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


def jarr(objs):
    import json
    return json.dumps(objs)


def main():
    case = load_intent_case(ROOT / "benchmark" / "cases" / "intent_hard")
    gold = case.gold
    checks = {}

    # ---- PHASE refine, both_cognitive: probe the two experiment-only intents, then submit ----
    t1 = _completion(tool_calls=[
        _call("c1", "check_feasibility", '{"intent_id":"I6","realisation_id":"r5a"}'),
        _call("c2", "check_feasibility", '{"intent_id":"I7","realisation_id":"r2b"}')])
    refinements = [
        {"intent_id": "I1", "realisation_id": "r1a", "satisfies": True},
        {"intent_id": "I2", "realisation_id": "", "satisfies": False},
        {"intent_id": "I3", "realisation_id": "", "satisfies": False},
        {"intent_id": "I4", "realisation_id": "r2b", "satisfies": True},
        {"intent_id": "I5", "realisation_id": "r3a", "satisfies": True},
        {"intent_id": "I6", "realisation_id": "r5a", "satisfies": True},
        {"intent_id": "I7", "realisation_id": "", "satisfies": False}]
    t2 = _completion(tool_calls=[_call("c3", "submit_refinement",
                                       '{"refinements":' + jarr(refinements) + '}')])
    rec = IntentAgentStack(case, model="fake", phase="refine",
                           client=FakeClient([t1, t2])).reconcile("both_cognitive")
    checks["refine submitted"] = rec.submitted
    checks["refine feasibility count==2"] = rec.oracle_calls["feasibility"] == 2
    # score against gold
    refine_ok = True
    for r in rec.refinements:
        g = gold["refine"][r["intent_id"]]
        want_sat = g["satisfiable"]
        if r["satisfies"] != want_sat:
            refine_ok = False
        if want_sat and r["realisation_id"] not in g["actual_satisfiers"]:
            refine_ok = False
    checks["refine matches gold"] = refine_ok and len(rec.refinements) == 7

    # ---- PHASE refine, one_inert (provider inert): feasibility refused ----
    t1b = _completion(tool_calls=[
        _call("c1", "check_feasibility", '{"intent_id":"I6","realisation_id":"r5a"}')])
    t2b = _completion(tool_calls=[_call("c2", "submit_refinement",
                                        '{"refinements":' + jarr(refinements) + '}')])
    rec_inert = IntentAgentStack(case, model="fake", phase="refine", inert_side="n",
                                 client=FakeClient([t1b, t2b])).reconcile("one_inert")
    refused = [s for s in rec_inert.transcript
               if s.get("step") == "check_feasibility" and not s["result"]["ok"]]
    checks["inert provider refuses feasibility"] = (len(refused) == 1
                                                    and rec_inert.oracle_calls["feasibility"] == 0)

    # ---- PHASE negotiate, both_cognitive, policy exec: obtain offers + judge, submit ----
    neg_intents = gold["negotiation_intents"]
    tn1 = _completion(tool_calls=[
        _call("n1", "best_achievable", '{"intent_id":"I2"}'),
        _call("n2", "consult_policy", '{"intent_id":"I2","realisation_id":"r1a"}'),
        _call("n3", "best_achievable", '{"intent_id":"I3"}'),
        _call("n4", "best_achievable", '{"intent_id":"I7"}')])
    decisions = []
    for iid in neg_intents:
        g = gold["negotiation"][iid]["exec"]
        dec = "reject" if g["decision"] == "reject" else "accept"
        decisions.append({"intent_id": iid, "offer_realisation": g["best_offer"], "decision": dec})
    tn2 = _completion(tool_calls=[_call("n5", "submit_negotiation",
                                        '{"decisions":' + jarr(decisions) + '}')])
    rec_neg = IntentAgentStack(case, model="fake", phase="negotiate", policy_id="exec",
                               client=FakeClient([tn1, tn2])).reconcile("both_cognitive")
    checks["negotiate submitted"] = rec_neg.submitted
    checks["negotiate best_achievable count==3"] = rec_neg.oracle_calls["best_achievable"] == 3
    neg_ok = True
    for d in rec_neg.decisions:
        g = gold["negotiation"][d["intent_id"]]["exec"]
        want = "reject" if g["decision"] == "reject" else "accept"
        if d["decision"] != want or d["offer_realisation"] != g["best_offer"]:
            neg_ok = False
    checks["negotiate matches gold (exec: all reject)"] = neg_ok

    # ---- PHASE negotiate, both_inert: policy + best_achievable refused ----
    ti1 = _completion(tool_calls=[
        _call("i1", "best_achievable", '{"intent_id":"I2"}'),
        _call("i2", "consult_policy", '{"intent_id":"I2","realisation_id":"r1a"}')])
    ti2 = _completion(tool_calls=[_call("i3", "submit_negotiation",
                                        '{"decisions":' + jarr(decisions) + '}')])
    rec_bi = IntentAgentStack(case, model="fake", phase="negotiate", policy_id="exec",
                              client=FakeClient([ti1, ti2])).reconcile("both_inert")
    bi_refused = all(not s["result"]["ok"] for s in rec_bi.transcript
                     if s.get("step") in ("best_achievable", "consult_policy"))
    checks["both_inert refuses best/policy"] = (bi_refused
                                                and rec_bi.oracle_calls["best_achievable"] == 0
                                                and rec_bi.oracle_calls["policy"] == 0)

    # ---- PHASE assure, both_cognitive, T1 set-piece: read telemetry per hop, submit ----
    ta1 = _completion(tool_calls=[
        _call("a1", "read_operational", '{"hop_id":"h0"}'),
        _call("a2", "read_operational", '{"hop_id":"h1"}'),
        _call("a3", "read_operational", '{"hop_id":"h2"}'),
        _call("a4", "read_operational", '{"hop_id":"h3"}')])
    tg = gold["lifecycle"]["T1"]["hops"]
    hops = [{"hop_id": h["hop_id"], "fulfilment": h["fulfilment"], "decision": h["decision"],
             "new_realisation": h["new_realisation"]} for h in tg]
    ta2 = _completion(tool_calls=[_call("a5", "submit_lifecycle", '{"hops":' + jarr(hops) + '}')])
    rec_as = IntentAgentStack(case, model="fake", phase="assure", policy_id="exec",
                              trajectory_id="T1", client=FakeClient([ta1, ta2])).reconcile("both_cognitive")
    checks["assure submitted"] = rec_as.submitted
    checks["assure telemetry count==4"] = rec_as.oracle_calls["telemetry"] == 4
    assure_ok = all(
        h["fulfilment"] == g["fulfilment"] and h["decision"] == g["decision"]
        and h["new_realisation"] == g["new_realisation"]
        for h, g in zip(rec_as.hops, tg))
    checks["assure matches gold T1 arc"] = assure_ok and len(rec_as.hops) == 4

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nT1 set-piece arc:",
          " -> ".join(f"{h['hop_id']}:{h['fulfilment']}/{h['decision']}->{h['new_realisation']}"
                      for h in rec_as.hops))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
