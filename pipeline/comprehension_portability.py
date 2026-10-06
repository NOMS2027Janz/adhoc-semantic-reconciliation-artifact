#!/usr/bin/env python3
"""SCAFFOLD (portability program, E22) -- destined for pipeline/comprehension_portability.py.

Comprehensibility of a lifted model to a cognitive CONSUMER, across the CONSUMER capability ladder, with
anchoring OFF vs ON. The keystone experiment of the portability program: a direct, consumer-relative
measure of lift quality, decoupled from reconciliation. See _shelf/PORTABILITY_program.md.

Arms (per case x model-side):
  * bare      -- the model's own surface only (labels, kinds, structure, instances, its own glosses),
                 no shared-reference anchoring.
  * anchored  -- + the shared reference (common-ground category each concept binds to). The bridge.
  [ later: constructed -- cognition self-anchors the single model (E25); a third arm. ]

Ladder = the CONSUMER model (sol / mini / nano). A fixed strong JUDGE grades the meaning answers.

Predictions: anchoring lifts meaning_score & use_recall CAPABILITY-INDEPENDENTLY (flat-high across the
consumer ladder); bare comprehension is capability-BANDED and confabulation_rate PEAKS at the mid tier;
a portable (anchored, well-lifted) model + a faithful consumer yields high ambiguity_flag_recall, low
over_claim.

Needs an authored comprehension gold per case: benchmark/cases/<case>/comprehension.json (see
_shelf/code/comprehension_gold.EXAMPLE.config_cross_domain.json). Two prompt blocks in comprehension.py are
DRAFT and the task set is a decision point -- confirm with Chris before running for real.

    # offline: check the gold format + scoring path for cases that have a comprehension.json (NO API)
    python pipeline/comprehension_portability.py --validate

    # the run (one consumer model per terminal = the consumer ladder):
    python pipeline/comprehension_portability.py --consumers gpt-5.6-sol --trials 3
    python pipeline/comprehension_portability.py --consumers gpt-5-mini  --trials 3
    python pipeline/comprehension_portability.py --consumers gpt-5-nano  --trials 3

Writes results/comprehension_portability.csv. Resumable. Needs OPENAI_API_KEY (except --validate).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                        # noqa: E402
from reconcile.comprehension import (load_comprehension, comprehend,     # noqa: E402
                                     judge_meaning, score_comprehension,
                                     ConsumerAnswer)

# Phase 1 reuses the hard seams already in play, so no new case construction is needed.
CASES = ["config_cross_domain", "config_big_hard", "config_rest"]
JUDGE_MODEL = "gpt-5.6-sol"     # fixed strong judge
ARMS = ["bare", "anchored"]     # + "constructed" (E25) later

COLS = ["case", "model_side", "arm", "consumer", "trial",
        "meaning_score", "confabulation_rate", "use_recall", "use_precision",
        "ambiguity_flag_recall", "over_claim", "total_tokens", "reasoning_tokens", "latency_s"]


def _side_model(case: Case, side: str):
    return case.model_a if side == "a" else case.model_b


def _payload(model, reference, *, anchored: bool) -> dict:
    """Bare vs anchored view of ONE model. Anchored adds each concept's shared-reference binding AND the
    reference entries themselves (id, label, class, definition, example) — the common-ground categories that
    make the model's local terms comprehensible to a consumer that has no insider knowledge of this system."""
    out = []
    for c in model.concepts:
        d = {"id": c.id, "label": c.label, "kind": c.kind, "synonyms": list(c.synonyms),
             "gloss": c.gloss, "example": c.example, "relations": list(c.relations),
             "instances": list(c.instances)}
        if anchored:
            d["ref"] = c.ref            # the anchoring: which shared-ground entry this binds to
        out.append(d)
    p = {"system": getattr(model, "system", ""), "dialect": getattr(model, "dialect", ""),
         "concepts": out}
    if anchored and reference is not None:
        p["reference"] = [{"id": e.id, "label": e.label, "class": e.cls,
                           "definition": e.definition, "example": e.example,
                           "synonyms": list(e.synonyms)} for e in reference.entries]
    return p


def _row(case, side, arm, consumer, t, scored, eff):
    return {"case": case, "model_side": side, "arm": arm, "consumer": consumer, "trial": t,
            **{k: scored.get(k, "") for k in
               ("meaning_score", "confabulation_rate", "use_recall", "use_precision",
                "ambiguity_flag_recall", "over_claim")},
            "total_tokens": eff.get("total_tokens", ""),
            "reasoning_tokens": eff.get("reasoning_tokens", ""),
            "latency_s": eff.get("latency_s", "")}


def _done_key(r):
    return (r["case"], r["model_side"], r["arm"], r["consumer"], str(r["trial"]))


def validate() -> int:
    """Offline: for every case that has a comprehension.json, load it and run the scoring path on a mock
    consumer answer. No API."""
    ok = True
    any_gold = False
    for case_name in CASES:
        cdir = ROOT / "benchmark" / "cases" / case_name
        if not (cdir / "comprehension.json").exists():
            print(f"  {case_name:20} (no comprehension.json yet -- author it)")
            continue
        any_gold = True
        gold = load_comprehension(cdir)
        # mock: consumer answers every question, marks the true relevant set, flags the true ambiguous set
        mock = ConsumerAnswer(
            meaning=[{"qid": q.id, "answer": q.answer_key} for q in gold.meaning_questions],
            use_relevant=list(gold.use_task.relevant_ids),
            flagged_ambiguous=list(gold.ambiguous_ids))
        verdicts = [{"qid": q.id, "label": "faithful"} for q in gold.meaning_questions]
        from reconcile.comprehension import Verdict
        s = score_comprehension(gold, mock, [Verdict(**v) for v in verdicts])
        good = (s["meaning_score"] == 1.0 and s["use_recall"] == 1.0)
        ok = ok and good
        print(f"  {case_name:20} model={gold.model} Qs={len(gold.meaning_questions)} "
              f"relevant={len(gold.use_task.relevant_ids)} ambiguous={len(gold.ambiguous_ids)} "
              f"-> mock score {'OK' if good else 'CHECK'} {s}")
    if not any_gold:
        print("  no comprehension.json authored yet -- this is the Phase-0 bottleneck.")
    print("validate:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consumers", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=str(ROOT / "results" / "comprehension_portability.csv"))
    ap.add_argument("--validate", action="store_true")
    # Cross-family isolation: point the CONSUMER at another provider while the JUDGE stays pinned to the
    # sol gateway. Passing consumer creds as FLAGS (not by exporting OPENAI_API_KEY) keeps the gateway key
    # in the environment for the judge, so consumer-family is the only thing that varies.
    ap.add_argument("--consumer-base-url", default=None,
                    help="OpenAI-compatible endpoint for the CONSUMER "
                         "(e.g. https://api.deepseek.com or http://localhost:11434/v1). "
                         "Default: OPENAI_BASE_URL, else the sol gateway.")
    ap.add_argument("--consumer-api-key", default=None,
                    help="API key for the consumer endpoint. Default: OPENAI_API_KEY.")
    ap.add_argument("--judge-base-url", default=None,
                    help="Endpoint for the fixed JUDGE. Default: OPENAI_JUDGE_BASE_URL, else the sol gateway.")
    ap.add_argument("--judge-api-key", default=None,
                    help="API key for the judge. Default: OPENAI_JUDGE_API_KEY, else OPENAI_API_KEY (gateway).")
    args = ap.parse_args()

    if args.validate:
        return validate()

    from run import load_dotenv  # noqa: E402
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    # Two clients: the consumer may live on another provider; the judge stays fixed on the sol gateway so
    # the ONLY thing that changes across a cross-family run is who does the comprehending.
    from openai import OpenAI  # noqa: E402

    consumer_base = args.consumer_base_url or os.environ.get("OPENAI_BASE_URL")
    consumer_key = args.consumer_api_key or os.environ.get("OPENAI_API_KEY")
    judge_base = args.judge_base_url or os.environ.get("OPENAI_JUDGE_BASE_URL")
    judge_key = (args.judge_api_key or os.environ.get("OPENAI_JUDGE_API_KEY")
                 or os.environ.get("OPENAI_API_KEY"))
    consumer_client = (OpenAI(base_url=consumer_base, api_key=consumer_key)
                       if consumer_base else OpenAI(api_key=consumer_key))
    judge_client = (OpenAI(base_url=judge_base, api_key=judge_key)
                    if judge_base else OpenAI(api_key=judge_key))
    print(f"consumer endpoint: {consumer_base or 'sol gateway (default)'}  |  "
          f"judge: {args.judge} @ {judge_base or 'sol gateway (default)'}", file=sys.stderr)

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
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

    # total planned units (runnable cases × valid arms × consumers × trials) for the [idx/total] index
    runnable = [c for c in cases if (ROOT / "benchmark" / "cases" / c / "comprehension.json").exists()]
    valid_arms = [a for a in arms if a in ("bare", "anchored")]
    total = len(runnable) * len(valid_arms) * len(consumers) * args.trials
    idx = 0
    n = 0
    for case_name in cases:
        cdir = ROOT / "benchmark" / "cases" / case_name
        if not (cdir / "comprehension.json").exists():
            print(f"  ! {case_name}: no comprehension.json -- skip (author it first)", file=sys.stderr)
            continue
        case = Case.load(cdir)
        gold = load_comprehension(cdir)
        model = _side_model(case, gold.model)
        for arm in arms:
            if arm not in ("bare", "anchored"):
                print(f"  ! arm {arm} not implemented in scaffold (constructed = E25 TODO)", file=sys.stderr)
                continue
            payload = _payload(model, case.reference, anchored=(arm == "anchored"))
            for consumer in consumers:
                for t in range(args.trials):
                    idx += 1
                    key = (case_name, gold.model, arm, consumer, str(t))
                    if key in seen:
                        continue
                    try:
                        answer, eff = comprehend(payload, gold, consumer, client=consumer_client)
                        verdicts = judge_meaning(gold, answer, args.judge, client=judge_client)
                        scored = score_comprehension(gold, answer, verdicts)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [{idx}/{total}] ! skip {case_name}/{arm}/{consumer}/t{t}: {e}", file=sys.stderr)
                        continue
                    row = _row(case_name, gold.model, arm, consumer, t, scored, eff)
                    w.writerow(row); fh.flush(); n += 1
                    print(f"  [{idx}/{total}] {case_name:20} {arm:9} {consumer:12} t{t}  "
                          f"mean={row['meaning_score']} conf={row['confabulation_rate']} "
                          f"useR={row['use_recall']} flag={row['ambiguity_flag_recall']}",
                          file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Readout: does anchoring lift meaning_score/use_recall at EVERY consumer tier "
          "(capability-independent portability), and does confabulation peak at the mid tier without it?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
