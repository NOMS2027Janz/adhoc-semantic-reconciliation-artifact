#!/usr/bin/env python3
"""Offline test (no API) of the observability Act-2 stacks and scoring, via scripted fakes."""
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from reconcile.observability import load_act2, verdict_of, correlate     # noqa: E402
from reconcile.stacks.agent_observability import (                        # noqa: E402
    VerdictStack, CorrelationStack, VerdictResult, _Verdict,
    CorrelationResult, _Incident)

CDIR = ROOT / "benchmark" / "cases" / "config_observability"


def _completion(parsed):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=20))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], usage=usage)


class FakeClient:
    def __init__(self, parsed):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=lambda **kw: _completion(parsed)))


def main():
    a2 = load_act2(CDIR)
    gold = json.loads((CDIR / "obs_act2_gold.json").read_text())
    anomalies, contexts, scenarios = a2["anomalies"], a2["contexts"], a2["scenarios"]
    checks = {}

    # --- verdict oracle self-consistency ---
    checks["oracle A1 flips act->suppress in maintenance"] = (
        verdict_of(next(a for a in anomalies if a["id"] == "A1"), {"id": "normal"}) == "act"
        and verdict_of(next(a for a in anomalies if a["id"] == "A1"),
                       {"id": "m", "maintenance_window": True}) == "suppress")

    # --- VerdictStack ON: perfect verdicts for the 'maintenance' context ---
    ctx_m = next(c for c in contexts if c["id"] == "maintenance")
    perfect = VerdictResult(verdicts=[_Verdict(anomaly_id=a["id"],
                                                verdict=gold["verdicts"][a["id"]]["maintenance"]) for a in anomalies])
    rec = VerdictStack(model="fake", pragmatics=True, client=FakeClient(perfect)).reconcile(anomalies, ctx_m)
    acc = sum(1 for a in anomalies if rec.verdicts.get(a["id"]) == gold["verdicts"][a["id"]]["maintenance"]) / len(anomalies)
    checks["verdict ON perfect acc==1.0"] = acc == 1.0
    checks["maintenance gold is all suppress"] = all(
        gold["verdicts"][a["id"]]["maintenance"] == "suppress" for a in anomalies)

    # --- VerdictStack OFF: legacy pages everything -> false pages under maintenance ---
    allact = VerdictResult(verdicts=[_Verdict(anomaly_id=a["id"], verdict="act") for a in anomalies])
    rec_off = VerdictStack(model="fake", pragmatics=False, client=FakeClient(allact)).reconcile(anomalies, ctx_m)
    false_pages = sum(1 for a in anomalies
                      if gold["verdicts"][a["id"]]["maintenance"] in ("suppress", "watch")
                      and rec_off.verdicts.get(a["id"]) == "act")
    checks["verdict OFF false-pages all under maintenance"] = false_pages == len(anomalies)

    # --- CorrelationStack ON: perfect incidents for S1 ---
    s1 = next(s for s in scenarios if s["id"] == "S1")
    g1 = gold["incidents"]["S1"]
    perfect_inc = CorrelationResult(incidents=[_Incident(symptoms=g["symptoms"], cause=g["cause"]) for g in g1])
    rec_c = CorrelationStack(model="fake", pragmatics=True, client=FakeClient(perfect_inc)).reconcile(s1)
    gparts = {frozenset(g["symptoms"]) for g in g1}
    aparts = {frozenset(a["symptoms"]) for a in rec_c.incidents}
    checks["correlation ON partition exact (S1)"] = aparts == gparts
    acause = {frozenset(a["symptoms"]): a["cause"] for a in rec_c.incidents}
    checks["correlation ON cause rooted optical (S1)"] = acause.get(frozenset(["s1", "s2"])) == "s1"

    # --- CorrelationStack OFF: one incident per symptom -> partition WRONG for S1 ---
    storm = CorrelationResult(incidents=[_Incident(symptoms=[s["id"]], cause=s["id"]) for s in s1["symptoms"]])
    rec_off_c = CorrelationStack(model="fake", pragmatics=False, client=FakeClient(storm)).reconcile(s1)
    aparts_off = {frozenset(a["symptoms"]) for a in rec_off_c.incidents}
    checks["correlation OFF is an alarm storm (S1 wrong)"] = (aparts_off != gparts
                                                              and len(rec_off_c.incidents) == 3)

    # --- pragmatic sensitivity present in the gold ---
    checks["pragmatic sensitivity present"] = len(gold["verdict_varies"]) >= 1

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
