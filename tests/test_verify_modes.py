#!/usr/bin/env python3
"""Offline test of the three verification modes (no API for the deterministic ones; a fake
client for invariant_round_trip). Asserts the mode-reach story the study is built to show:

  * byte round-trip passes everything -> catches no wrong pair (false-pass 1.0).
  * virtual operation catches the byte-clean crossed pairs where a live side exists, and its
    reach collapses as cognition recedes.
  * invariant round-trip catches the meaning-visible wrong pairs but is blind to byte-clean.
"""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.instance import load_instance_case, Oracle          # noqa: E402
from reconcile import verify_modes as VM                           # noqa: E402

# reuse the driver's scoring
sys.path.insert(0, str(ROOT / "pipeline"))
import verify_study as VS                                          # noqa: E402


class FakeParse:
    """A stand-in invariant verifier: fails a proposal when the two records differ in their
    stable signature (type, attributes, topology degree), passes otherwise. This is what an
    ideal record-only invariant verifier does — it catches meaning-visible differences and is
    blind to byte-clean identical records."""
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _sig(self, rec):
        return (rec["type"], tuple(sorted((rec.get("attrs") or {}).items())),
                len(rec.get("topology") or []))

    def _parse(self, model, messages, response_format):
        payload = json.loads(messages[1]["content"])
        verdicts = []
        for p in payload["proposals"]:
            passes = self._sig(p["a"]) == self._sig(p["b"])
            verdicts.append(SimpleNamespace(id=p["id"], passes=passes, reason=""))
        parsed = SimpleNamespace(verdicts=verdicts)
        usage = SimpleNamespace(total_tokens=200,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=50))
        msg = SimpleNamespace(parsed=parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def main():
    proposals, gold = VS.load_verify(ROOT / "benchmark" / "cases" / "verify_hard")
    case = load_instance_case(ROOT / "benchmark" / "cases" / "instance_hard")
    inv = proposals.get("invariants", [])
    checks = {}

    # byte
    bv = {p["id"]: VM.byte_round_trip(p) for p in proposals["proposals"]}
    bs = VS.score(bv, proposals, gold)
    checks["byte catches nothing"] = bs["catch_rate"] == 0.0
    checks["byte false-pass 1.0"] = bs["false_pass_rate"] == 1.0
    checks["byte reach 1.0"] = bs["reach"] == 1.0

    # virtual, three placements
    vs_bc = VS.score({p["id"]: VM.virtual_operation(p, Oracle(case, "both_cognitive", budget=None))
                      for p in proposals["proposals"]}, proposals, gold)
    vs_oi = VS.score({p["id"]: VM.virtual_operation(p, Oracle(case, "one_inert", budget=None))
                      for p in proposals["proposals"]}, proposals, gold)
    vs_bi = VS.score({p["id"]: VM.virtual_operation(p, Oracle(case, "both_inert", budget=None))
                      for p in proposals["proposals"]}, proposals, gold)
    checks["virtual both_cog catches byte-clean (bc=1.0)"] = vs_bc["bc_catch_rate"] == 1.0
    checks["virtual both_cog no false-pass"] = vs_bc["false_pass_rate"] == 0.0
    checks["virtual both_cog catch>=0.8"] = vs_bc["catch_rate"] >= 0.8
    checks["virtual one_inert loses byte-clean (bc=0)"] = vs_oi["bc_catch_rate"] == 0.0
    checks["virtual one_inert still catches svc (mv>0)"] = (vs_oi["mv_catch_rate"] or 0) > 0
    checks["virtual both_inert reach 0"] = vs_bi["reach"] == 0.0

    # invariant round-trip (fake), both_cognitive
    fv, eff = VM.invariant_round_trip(proposals["proposals"], "both_cognitive", "fake", inv,
                                      client=FakeParse())
    iscore = VS.score(fv, proposals, gold, eff)
    checks["invariant catches meaning-visible (mv=1.0)"] = iscore["mv_catch_rate"] == 1.0
    checks["invariant blind to byte-clean (bc=0.0)"] = iscore["bc_catch_rate"] == 0.0
    checks["invariant passes all correct"] = iscore["fail_correct"] == 0

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nbyte:      catch {bs['catch_rate']} false-pass {bs['false_pass_rate']} reach {bs['reach']}")
    print(f"virtual bc: catch {vs_bc['catch_rate']} bc {vs_bc['bc_catch_rate']} mv {vs_bc['mv_catch_rate']} reach {vs_bc['reach']}")
    print(f"virtual oi: catch {vs_oi['catch_rate']} bc {vs_oi['bc_catch_rate']} mv {vs_oi['mv_catch_rate']} reach {vs_oi['reach']}")
    print(f"virtual bi: catch {vs_bi['catch_rate']} reach {vs_bi['reach']}")
    print(f"invariant:  catch {iscore['catch_rate']} mv {iscore['mv_catch_rate']} bc {iscore['bc_catch_rate']} false-pass {iscore['false_pass_rate']}")
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
