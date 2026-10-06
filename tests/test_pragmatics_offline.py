#!/usr/bin/env python3
"""Offline test of the pragmatics stack + scorer (no API), via a scripted fake client."""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                    # noqa: E402
from reconcile.stacks.agent_pragmatics import (                     # noqa: E402
    PragmaticsStack, PragmaticsResult, _Attribution)
import pragmatics_study as P                                        # noqa: E402


def fake_completion(attributions):
    result = PragmaticsResult(attributions=[_Attribution(**a) for a in attributions])
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=80, total_tokens=280,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))],
                           usage=usage)


class FakeClient:
    def __init__(self, attributions):
        self._a = attributions
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=lambda **kw: fake_completion(self._a)))


def main():
    case_dir = ROOT / "benchmark" / "cases" / "config_cross_domain"
    case = Case.load(case_dir)
    fields = json.loads((case_dir / "pragmatics.json").read_text())["fields"]
    checks = {}

    # 1) perfect authority attribution -> accuracy 1.0
    perfect = [{"field_id": f["id"], "authority": f["authority"], "rationale": "x"} for f in fields]
    rec = PragmaticsStack(case, fields, use_reference=True, model="fake",
                          client=FakeClient(perfect)).reconcile("both_cognitive")
    s = P.score(rec, fields)
    checks["perfect acc==1.0"] = s["authority_accuracy"] == 1.0
    checks["perfect over-errors==0"] = (s["over_transport"] == 0 and s["over_ip"] == 0
                                        and s["shared_missed"] == 0)
    checks["submitted"] = rec.submitted

    # 2) transport-biased: answer X for everything (the 'transport owns all it carries' failure)
    allX = [{"field_id": f["id"], "authority": "X", "rationale": "x"} for f in fields]
    rec2 = PragmaticsStack(case, fields, use_reference=False, model="fake",
                           client=FakeClient(allX)).reconcile("both_inert")
    s2 = P.score(rec2, fields)
    # gold: 2 X, 2 Y, 1 shared -> answering all X gets the 2 X right
    checks["allX acc==0.4"] = s2["authority_accuracy"] == 0.4
    checks["allX over_transport==3"] = s2["over_transport"] == 3
    checks["allX shared_missed==1"] = s2["shared_missed"] == 1

    # 3) case masking: inert side hides gloss/example in the payload
    stack = PragmaticsStack(case, fields, use_reference=False, model="fake", client=FakeClient(perfect))
    payload_bc = json.loads(stack._payload("both_cognitive"))
    payload_oi = json.loads(stack._payload("one_inert"))
    y_bc = payload_bc["agent_Y_cascade"]["concepts"][0]
    y_oi = payload_oi["agent_Y_cascade"]["concepts"][0]
    checks["live Y has gloss"] = "gloss" in y_bc
    checks["inert Y hides gloss"] = "gloss" not in y_oi and payload_oi["agent_Y_cascade"]["inert"] is True
    checks["reference omitted when no-ref"] = "ad_hoc_reference" not in payload_bc

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("perfect fields:", s["fields"])
    print("allX fields:   ", s2["fields"])
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
