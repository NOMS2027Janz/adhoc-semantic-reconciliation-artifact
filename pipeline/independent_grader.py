#!/usr/bin/env python3
"""E10b -- an independent grader: does a different judge agree with the derived gold? (Brad pt 11.)

The study scores reconciliation against DERIVED gold. Brad's point 11 also asks for a second,
independently-implemented grader -- a different judge -- to check the findings are not gold- or
grader-specific. This driver builds a candidate set from a case's gold correspondences (should judge
SAME), its planted false cognates (should judge NOT same), and decoy non-pairs (NOT same), and asks a
JUDGE MODEL to rule same/not-same on each pair from the two concepts' descriptions alone -- never
shown the gold. Comparing the judge to the gold tests two things at once:

  * is the derived gold TRUSTWORTHY -- does an independent judge confirm the true correspondences and
    reject the planted false cognates? (agreement = the gold is grader-independent);
  * is scoring FAMILY-dependent -- run the judge as a gpt model AND as a local open-weights model
    (via OPENAI_BASE_URL) and compare.

    python pipeline/independent_grader.py --case config_evpn --judge-model gpt-5.6-sol --trials 2
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=local \
        python pipeline/independent_grader.py --case config_evpn --judge-model qwen2.5:7b --trials 2

Writes results/independent_grader_<case>.csv (one row per judge x pair x trial).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile import Case   # noqa: E402
from run import load_dotenv  # noqa: E402

COLS = ["case", "judge_model", "pair", "category", "trial",
        "expected_same", "judged_same", "correct", "confidence"]

JUDGE_SYSTEM = (
    "You judge whether two concepts, each from a different network model, denote the SAME thing (a "
    "true correspondence) or merely look alike (a false cognate). Decide by MEANING -- the gloss, "
    "examples and kind -- not by the label: two concepts can share a word and denote different "
    "things, or use different words for the same thing. Answer same=true only if they denote the "
    "same underlying thing. You are NOT given any answer key; judge from the descriptions."
)


class _Verdict(BaseModel):
    same: bool
    confidence: float
    reason: str


def _cview(c):
    return {"label": c.label, "kind": c.kind, "gloss": c.gloss,
            "synonyms": list(c.synonyms), "example": c.example}


def build_candidates(case):
    a_by = {c.id: c for c in case.model_a.concepts}
    b_by = {c.id: c for c in case.model_b.concepts}
    gold = {(g["a"], g["b"]) for g in case.gold.correspondences}
    traps = {(t["a"], t["b"]) for t in case.gold.false_cognates}
    cands = []
    for (a, b) in gold:
        if a in a_by and b in b_by:
            cands.append((a, b, "gold", True))
    for (a, b) in traps:
        if a in a_by and b in b_by:
            cands.append((a, b, "trap", False))
    # decoys: pair each a-concept with a b-concept that is neither its gold nor trap partner
    b_ids = [c.id for c in case.model_b.concepts]
    used = gold | traps
    for i, ca in enumerate(case.model_a.concepts):
        b = b_ids[(i + 3) % len(b_ids)]          # deterministic shift
        if (ca.id, b) not in used and ca.id in a_by and b in b_by:
            cands.append((ca.id, b, "decoy", False))
    return cands, a_by, b_by


def judge_pair(a_view, b_view, model, client):
    from reconcile.compat import parse_compat
    completion = parse_compat(
        client, model,
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": json.dumps({"concept_A": a_view, "concept_B": b_view}, indent=2)}],
        _Verdict,
    )
    return completion.choices[0].message.parsed


def run_key(r):
    return (str(r["judge_model"]), r["pair"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_evpn")
    ap.add_argument("--judge-model", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    cands, a_by, b_by = build_candidates(case)
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from openai import OpenAI
    base = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(base_url=base) if base else OpenAI()
    print(f"judge: {args.judge_model}  endpoint: {base or '(OpenAI)'}  "
          f"candidates: {sum(1 for c in cands if c[2]=='gold')} gold / "
          f"{sum(1 for c in cands if c[2]=='trap')} trap / {sum(1 for c in cands if c[2]=='decoy')} decoy",
          file=sys.stderr)

    out = ROOT / "results" / f"independent_grader_{args.case}.csv"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)
    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        rows = list(csv.DictReader(out.open(newline="")))
        done = {run_key(r) for r in rows}
        print(f"Resuming: {len(done)} rows in {out.name}", file=sys.stderr)
    fh = writer = None
    if write:
        fresh = args.fresh or not out.exists()
        fh = out.open("w" if fresh else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLS)
        if fresh:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    def emit(row):
        full = {c: row.get(c, "") for c in COLS}
        rows.append(full); done.add(run_key(full))
        if writer is not None:
            writer.writerow(full); fh.flush()

    total = len(cands) * trials
    i = 0
    for (a, b, cat, expected) in cands:
        pair = f"{a}~{b}"
        for t in range(trials):
            i += 1
            if (str(args.judge_model), pair, str(t)) in done:
                continue
            print(f"[{i}/{total}] {args.judge_model} {pair} ({cat}) t{t+1}", file=sys.stderr, flush=True)
            try:
                v = judge_pair(_cview(a_by[a]), _cview(b_by[b]), args.judge_model, client)
            except Exception as e:  # noqa: BLE001
                print(f"      ! {e}", file=sys.stderr, flush=True)
                continue
            js = bool(v.same) if v else False
            emit({"case": args.case, "judge_model": args.judge_model, "pair": pair, "category": cat,
                  "trial": t, "expected_same": expected, "judged_same": js,
                  "correct": int(js == expected), "confidence": (v.confidence if v else "")})

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    if not rows:
        return
    judges = []
    for r in rows:
        if r["judge_model"] not in judges:
            judges.append(r["judge_model"])

    def rate(rs):
        xs = [1.0 if str(r["correct"]) == "1" else 0.0 for r in rs]
        return sum(xs) / len(xs) if xs else None

    print("\n=== E10b  Independent grader vs the derived gold ===")
    print(f"    {'judge':14}{'overall':9}{'gold-confirm':14}{'trap-reject':13}{'decoy-reject':13}")
    for j in judges:
        jr = [r for r in rows if r["judge_model"] == j]
        g = rate([r for r in jr if r["category"] == "gold"])
        tp = rate([r for r in jr if r["category"] == "trap"])
        d = rate([r for r in jr if r["category"] == "decoy"])
        def s(v):
            return f"{v:.2f}" if v is not None else "-"
        print(f"    {j:14}{s(rate(jr)):9}{s(g):14}{s(tp):13}{s(d):13}")
    print("\n    Reading: high gold-confirm + high trap-reject + high decoy-reject = the independent")
    print("    judge agrees with the derived gold (the gold is grader-independent). Compare a gpt")
    print("    judge with a local open-weights judge to test whether scoring is family-dependent.")


if __name__ == "__main__":
    raise SystemExit(main())
