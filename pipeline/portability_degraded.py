#!/usr/bin/env python3
"""E22 Phase 3 — degrade the LIFT, then measure comprehensibility (portability programme).

Produce a model by lifting a THINNED surface (+/- reference during lifting), then comprehend it across the
reasoning ladder. Tests the production-side role of the reference and the confabulation hazard. See
_shelf/PORTABILITY_program.md "E22 Phase 3".

Pipeline: thin_model (mirror of E17 cold_start_lift.thin) -> lift_model(reference=...) -> comprehension.

THREE reference cases (the 4th, ref-not-in-lift-but-at-consumption, is incoherent and omitted):
  case1  no-ref       : lift WITHOUT reference; consume bare.
  case2  lift-only    : lift WITH reference (meaning dissolved into glosses); consume bare (self-sufficient?).
  case3  lift+consume : lift WITH reference; consume WITH reference glossary too (ceiling).

Fixed lifter = sol (degradation comes from the SURFACE, not a weak lifter). Rungs = named (rich control) /
typed / data. Metrics (no "recall" per the study's terminology): meaning_score (faithful), abstention_rate
(honest "cannot determine" — a VIRTUE on a degraded model), confabulation_rate (invented — the hazard).

Concept ids are DESCRIPTIVE (m.circuit), so thinning makes them opaque (c1,c2) to stop the consumer reading
meaning off the id; the gold's questions are translated to those opaque ids for the consumer and scored back.

    python pipeline/portability_degraded.py --validate            # offline: thinning + scoring, no API
    python pipeline/portability_degraded.py --consumers gpt-5.6-sol,gpt-5-mini,gpt-5-nano --rungs named,typed,data --trials 3
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case, SemanticModel                          # noqa: E402
from reconcile.lift import lift_model                                    # noqa: E402
from reconcile.comprehension import (load_comprehension, Comprehension, MeaningQ,  # noqa: E402
                                     comprehend, Verdict, JudgeResult, _parse)
from comprehension_portability import _payload                           # noqa: E402

CASES = ["config_cross_domain", "config_big_hard", "config_rest"]
RUNGS = ["named", "typed", "data"]           # E17 ladder minus topology (too degenerate)
LIFTER = "gpt-5.6-sol"                        # fixed, so degradation is the SURFACE not the lifter
JUDGE_MODEL = "gpt-5.6-sol"
# (label, ref_in_lift, consume_anchored)
REF_CASES = [("no-ref", False, False), ("lift-only", True, False), ("lift+consume", True, True)]

COLS = ["case", "rung", "ref_case", "lifter", "consumer", "trial",
        "meaning_score", "abstention_rate", "confabulation_rate", "lift_coverage",
        "total_tokens", "reasoning_tokens"]


# ---- thin a MODEL, faithful to cold_start_lift.thin() (kept kind at typed; stripped at data) --------
def thin_model(sm: SemanticModel, rung: str):
    """Return (thinned SemanticModel, idmap opaque->real). Mirrors E17 thin(): opaque ids and labels when
    names are stripped, synonyms dropped, relations genericized to opaque targets, kind kept at typed /
    'leaf' at data, instances only where data survives. gloss/example always blank (never handed to the lift)."""
    opaque_of = {c.id: f"c{i+1}" for i, c in enumerate(sm.concepts)}
    names = rung == "named"
    types = rung in ("named", "typed")
    data = rung in ("named", "typed", "data")
    idmap, new = {}, []
    for c in sm.concepts:
        pid = c.id if names else opaque_of[c.id]
        idmap[pid] = c.id
        if names:
            rels = [dict(r) for r in c.relations]
        else:
            rels = [{"rel": "ref", "target": opaque_of.get(r.get("target"), "external")} for r in c.relations]
        new.append(dataclasses.replace(
            c, id=pid,
            label=(c.label if names else pid),
            synonyms=(list(c.synonyms) if names else []),
            kind=(c.kind if types else "leaf"),
            relations=rels,
            instances=(list(c.instances) if data else []),
            gloss="", example="", ref=None,
        ))
    return SemanticModel(system=sm.system, dialect=sm.dialect, modules=sm.modules, concepts=new), idmap


def _remap_gold(gold: Comprehension, real_to_opaque: dict) -> Comprehension:
    """Translate each meaning question's concept_id to the OPAQUE id the consumer will see (identity at the
    named rung). qid and answer_key are unchanged, so scoring is unaffected."""
    qs = [dataclasses.replace(q, concept_id=real_to_opaque.get(q.concept_id, q.concept_id))
          for q in gold.meaning_questions]
    return dataclasses.replace(gold, meaning_questions=qs)


# ---- abstained-aware judge + scoring (no "recall") --------------------------------------------------
_JUDGE_SYSTEM_P3 = (
    "You grade a consumer's stated meaning for each concept against an authored answer key, over a possibly "
    "DEGRADED model. For each question return a verdict: 'faithful' (matches the key), 'partial' (right "
    "direction, hedged/incomplete), 'abstained' (the consumer honestly says it cannot determine the meaning "
    "from what it was given — NOT a guess), or 'invented' (asserts a meaning the key does not support). "
    "Abstaining when the meaning is genuinely absent is correct, not a failure; inventing is the failure."
)


def judge_p3(gold: Comprehension, answer, judge_model: str, client=None):
    by = {a.qid: a.answer for a in answer.meaning}
    payload = {"items": [{"qid": q.id, "answer_key": q.answer_key,
                          "must_not_confuse_with": q.must_not_confuse_with,
                          "consumer_answer": by.get(q.id, "")} for q in gold.meaning_questions]}
    res, _eff = _parse(client, judge_model, _JUDGE_SYSTEM_P3, payload, JudgeResult)
    return res.verdicts


def score_p3(gold: Comprehension, verdicts) -> dict:
    v = {x.qid: x.label for x in verdicts}
    n = max(1, len(gold.meaning_questions))
    cnt = lambda lab: sum(1 for q in gold.meaning_questions if v.get(q.id) == lab)
    return {"meaning_score": round(cnt("faithful") / n, 3),
            "abstention_rate": round(cnt("abstained") / n, 3),
            "confabulation_rate": round(cnt("invented") / n, 3)}


def validate() -> int:
    ok = True
    for case_name in CASES:
        cdir = ROOT / "benchmark" / "cases" / case_name
        if not (cdir / "comprehension.json").exists():
            print(f"  {case_name:20} (no comprehension.json)"); continue
        case = Case.load(cdir); n = len(case.model_a.concepts)
        for rung in RUNGS:
            thinned, idmap = thin_model(case.model_a, rung)
            r2o = {v: k for k, v in idmap.items()}
            gold = _remap_gold(load_comprehension(cdir), r2o)
            back = {idmap[q.concept_id] for q in gold.meaning_questions}
            orig = {q.concept_id for q in load_comprehension(cdir).meaning_questions}
            same = len(thinned.concepts) == n and back == orig
            ok = ok and same
            c0 = thinned.concepts[0]
            print(f"  {case_name:18} {rung:6} n={len(thinned.concepts)} id0={c0.id} label={c0.label!r} "
                  f"kind={c0.kind!r} inst={len(c0.instances)} q-roundtrip={back==orig}")
    print("validate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _done_key(r):
    return (r["case"], r["rung"], r["ref_case"], r["consumer"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lifter", default=LIFTER)
    ap.add_argument("--consumers", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "portability_degraded.csv"))
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        return validate()

    from run import load_dotenv  # noqa: E402
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr); return 2

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    consumers = [m.strip() for m in args.consumers.split(",") if m.strip()]
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add(_done_key(r))
    new = not out.exists()
    fh = out.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    from openai import OpenAI
    client = OpenAI()
    # total planned units, counting only cases that will actually run — for the [idx/total] progress index
    runnable = [c for c in cases if (ROOT / "benchmark" / "cases" / c / "comprehension.json").exists()]
    total = len(runnable) * len(rungs) * args.trials * len(REF_CASES) * len(consumers)
    idx = 0
    n = 0
    for case_name in cases:
        cdir = ROOT / "benchmark" / "cases" / case_name
        if not (cdir / "comprehension.json").exists():
            print(f"  ! {case_name}: no comprehension.json — skip", file=sys.stderr); continue
        case = Case.load(cdir)
        gold_real = load_comprehension(cdir)
        for rung in rungs:
            for t in range(args.trials):
                # lift the thinned surface once per (rung, ref_in_lift, trial); cache for all consumers
                lift_cache = {}   # ref_in_lift -> (lifted_model, idmap, lift_eff)
                for ref_label, ref_in_lift, consume_anchored in REF_CASES:
                    # skip early if every consumer row for this ref_case is already done
                    if all((case_name, rung, ref_label, cm, str(t)) in seen for cm in consumers):
                        idx += len(consumers)   # keep the index aligned across resumes
                        continue
                    if ref_in_lift not in lift_cache:
                        thinned, idmap = thin_model(case.model_a, rung)
                        try:
                            lifted, leff = lift_model(thinned, args.lifter, client=client,
                                                      reference=(case.reference if ref_in_lift else None))
                        except Exception as e:  # noqa: BLE001
                            print(f"  ! lift fail {case_name}/{rung}/rin={ref_in_lift}/t{t}: {e}", file=sys.stderr)
                            continue
                        lift_cache[ref_in_lift] = (lifted, idmap, leff)
                    lifted, idmap, leff = lift_cache[ref_in_lift]
                    r2o = {v: k for k, v in idmap.items()}
                    gold = _remap_gold(gold_real, r2o)
                    payload = _payload(lifted, case.reference, anchored=consume_anchored)
                    for cm in consumers:
                        idx += 1
                        key = (case_name, rung, ref_label, cm, str(t))
                        if key in seen:
                            continue
                        try:
                            answer, ceff = comprehend(payload, gold, cm, client=client)
                            verdicts = judge_p3(gold, answer, args.judge, client=client)
                            scored = score_p3(gold, verdicts)
                        except Exception as e:  # noqa: BLE001
                            print(f"  [{idx}/{total}] ! skip {case_name}/{rung}/{ref_label}/{cm}/t{t}: {e}",
                                  file=sys.stderr)
                            continue
                        row = {"case": case_name, "rung": rung, "ref_case": ref_label, "lifter": args.lifter,
                               "consumer": cm, "trial": t, **scored,
                               "lift_coverage": leff.get("lift_coverage", ""),
                               "total_tokens": ceff.get("total_tokens", ""),
                               "reasoning_tokens": ceff.get("reasoning_tokens", "")}
                        w.writerow(row); fh.flush(); n += 1
                        print(f"  [{idx}/{total}] {case_name:18} {rung:6} {ref_label:12} {cm:12} t{t}  "
                              f"mean={row['meaning_score']} abst={row['abstention_rate']} "
                              f"conf={row['confabulation_rate']} cov={row['lift_coverage']}", file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Readout: does meaning_score fall as the surface thins (no-ref), does lift-only (case 2) restore it "
          "toward the named control, and does confabulation rise at the mid/weak tiers where honest abstention should?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
