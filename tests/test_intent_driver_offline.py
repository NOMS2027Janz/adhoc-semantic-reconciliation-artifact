#!/usr/bin/env python3
"""Offline end-to-end check of intent_study.py's four phases (no API).

Monkeypatches the agent stacks' client factory to a 'perfect agent' that submits the gold
in one turn, then runs each phase with a fake model and asserts the driver plans, scores,
writes segmented CSVs, and (phase 4) retains transcripts. Cleans up its fake result files.
"""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

import intent_study as S                                              # noqa: E402
from reconcile.stacks.agent_intent import IntentAgentStack, SUBMIT_NAME  # noqa: E402
from reconcile.stacks.agent_instance import InstanceAgentStack        # noqa: E402


def _usage():
    return SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150,
                           output_tokens_details=SimpleNamespace(reasoning_tokens=20))


def _submit(name, args):
    return SimpleNamespace(output=[{"type": "function_call", "call_id": "s0", "name": name,
                                    "arguments": json.dumps(args)}], usage=_usage())


class OneShot:
    def __init__(self, resp):
        self.responses = SimpleNamespace(create=lambda **kw: resp)


def perfect_intent_client(self):
    g = self.case.gold
    if self.phase == "refine":
        refs = []
        for iid, gg in g["refine"].items():
            rid = gg["actual_satisfiers"][0] if gg["satisfiable"] else ""
            refs.append({"intent_id": iid, "realisation_id": rid, "satisfies": gg["satisfiable"]})
        args = {"refinements": refs}
    elif self.phase == "negotiate":
        decs = []
        for iid in g["negotiation_intents"]:
            gg = g["negotiation"][iid][self.policy_id]
            decs.append({"intent_id": iid, "offer_realisation": gg["best_offer"],
                         "decision": gg["decision"]})
        args = {"decisions": decs}
    else:
        hops = [{"hop_id": h["hop_id"], "fulfilment": h["fulfilment"], "decision": h["decision"],
                 "new_realisation": h["new_realisation"]}
                for h in g["lifecycle"][self.trajectory_id]["hops"]]
        args = {"hops": hops}
    return OneShot(_submit(SUBMIT_NAME[self.phase], args))


def perfect_instance_client(self):
    corr = [{"a_id": c["a"], "b_id": c["b"], "confidence": 0.9}
            for c in self.case.gold["correspondences"]]
    return OneShot(_submit("submit_alignment",
                           {"correspondences": corr, "residual_a": [], "residual_b": []}))


def main():
    IntentAgentStack._get_client = perfect_intent_client
    InstanceAgentStack._get_client = perfect_instance_client
    case = S.load_intent_case(ROOT / "benchmark" / "cases" / "intent_hard")
    models = ["fake"]
    checks = {}

    r1 = S.phase1(case, models, trials=1, budget=20, fresh=True, write=True)
    checks["phase1 rows"] = len(r1) == len(S.SPECTRUM_SIMPLE) * len(S.REFERENCES)
    checks["phase1 perfect sat-acc"] = all(float(r["satisfaction_accuracy"]) == 1.0 for r in r1)

    r2 = S.phase2(case, models, trials=1, budget=20, fresh=True, write=True)
    n_plan2 = len(case.policies) * (len(S.SPECTRUM_FULL) + 2)
    checks["phase2 rows"] = len(r2) == n_plan2
    checks["phase2 perfect dec-acc"] = all(float(r["decision_accuracy"]) == 1.0 for r in r2)

    r3 = S.phase3(models, trials=1, budget=20, fresh=True, write=True)
    checks["phase3 rows"] = len(r3) == len(S.SPECTRUM_SIMPLE)
    checks["phase3 perfect recall"] = all(float(r["recall"]) == 1.0 for r in r3)

    r4 = S.phase4(case, models, trials=1, budget=20, fresh=True, write=True)
    checks["phase4 rows"] = len(r4) == len(case.lifecycle)
    checks["phase4 perfect hop-acc"] = all(float(r["hop_accuracy"]) == 1.0 for r in r4)
    checks["phase4 setpiece transcript saved"] = (
        ROOT / "results" / "intent_transcripts" / "T1_fake.json").exists()

    # resume: re-run phase1, should skip all (no new rows appended beyond the plan size)
    r1b = S.phase1(case, models, trials=1, budget=20, fresh=False, write=True)
    checks["phase1 resume stable"] = len(r1b) == len(r1)

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    # cleanup fake artefacts
    for p in (ROOT / "results").glob("intent_*_fake.csv"):
        p.unlink()
    for p in (ROOT / "results").glob("intent_endpoints_phase3_fake.csv"):
        p.unlink()
    tp = ROOT / "results" / "intent_transcripts" / "T1_fake.json"
    for p in (ROOT / "results" / "intent_transcripts").glob("*_fake.json"):
        p.unlink()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
