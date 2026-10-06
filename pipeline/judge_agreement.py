#!/usr/bin/env python3
"""Paired judge-agreement: is the comprehension meaning-metric judge-independent?

The portability runs grade every consumer answer with one fixed judge (sol). This driver closes the
"house-judge" flank the airtight way: it generates each consumer answer ONCE, persists it, then grades
that SAME answer with TWO judges (default: sol on the gateway, and a cross-family judge such as
deepseek-chat). Because the answer is frozen, the only thing that differs between the two gradings is
the judge, so per-item disagreement is pure judge variance with zero consumer-generation noise.

Reports per-item agreement and Cohen's kappa over the verdict labels, plus each judge's meaning_score,
and writes the frozen answers + both judges' verdicts to a JSON for the record.

    # sol (gateway) vs deepseek-chat as the second judge:
    python pipeline/judge_agreement.py --case config_cross_domain \
        --consumers gpt-5.6-sol,gpt-5-nano --trials 3 \
        --judge-b deepseek-chat --judge-b-base-url https://api.deepseek.com --judge-b-api-key "$DEEPSEEK_KEY"

Writes results/judge_agreement_<case>.csv (one row per consumer x arm x trial x question) and
results/judge_agreement_<case>.json. Consumer + judge-A run on the gateway; only judge-B is redirected.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                    # noqa: E402
from reconcile.comprehension import load_comprehension, comprehend, judge_meaning  # noqa: E402
from comprehension_portability import _payload, _side_model          # noqa: E402

COLS = ["case", "consumer", "arm", "trial", "qid",
        "judge_a", "judge_b", "label_a", "label_b", "agree"]


def _cohens_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns Cohen's kappa over the observed label set."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted({x for p in pairs for x in p})
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_cross_domain")
    ap.add_argument("--consumers", default="gpt-5.6-sol,gpt-5-nano")
    ap.add_argument("--arms", default="bare,anchored")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--judge-a", default="gpt-5.6-sol")
    ap.add_argument("--judge-b", default="deepseek-chat")
    ap.add_argument("--judge-b-base-url", default=None)
    ap.add_argument("--judge-b-api-key", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from run import load_dotenv  # noqa: E402
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    from openai import OpenAI
    gateway = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))          # consumer + judge A
    jb_base = args.judge_b_base_url or os.environ.get("OPENAI_JUDGE_BASE_URL")
    jb_key = (args.judge_b_api_key or os.environ.get("OPENAI_JUDGE_API_KEY")
              or os.environ.get("OPENAI_API_KEY"))
    judge_b_client = OpenAI(base_url=jb_base, api_key=jb_key) if jb_base else OpenAI(api_key=jb_key)
    print(f"consumer + judge A ({args.judge_a}): sol gateway  |  "
          f"judge B ({args.judge_b}): {jb_base or 'sol gateway'}", file=sys.stderr)

    cdir = ROOT / "benchmark" / "cases" / args.case
    case = Case.load(cdir)
    gold = load_comprehension(cdir)
    model = _side_model(case, gold.model)
    consumers = [c.strip() for c in args.consumers.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    out = Path(args.out) if args.out else ROOT / "results" / f"judge_agreement_{args.case}.csv"
    out.parent.mkdir(exist_ok=True)
    fh = out.open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()

    pairs, ma, mb, record = [], [], [], []
    total = len(consumers) * len(arms) * args.trials
    idx = 0
    for arm in arms:
        payload = _payload(model, case.reference, anchored=(arm == "anchored"))
        for consumer in consumers:
            for t in range(args.trials):
                idx += 1
                try:
                    answer, _eff = comprehend(payload, gold, consumer, client=gateway)   # FROZEN answer
                    va = judge_meaning(gold, answer, args.judge_a, client=gateway)
                    vb = judge_meaning(gold, answer, args.judge_b, client=judge_b_client)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{idx}/{total}] ! skip {arm}/{consumer}/t{t}: {e}", file=sys.stderr)
                    continue
                la = {x.qid: x.label for x in va}
                lb = {x.qid: x.label for x in vb}
                for q in gold.meaning_questions:
                    a_lab, b_lab = la.get(q.id, ""), lb.get(q.id, "")
                    if a_lab and b_lab:
                        pairs.append((a_lab, b_lab))
                    w.writerow({"case": args.case, "consumer": consumer, "arm": arm, "trial": t,
                                "qid": q.id, "judge_a": args.judge_a, "judge_b": args.judge_b,
                                "label_a": a_lab, "label_b": b_lab, "agree": int(a_lab == b_lab)})
                fh.flush()
                ma.append(sum(1 for q in gold.meaning_questions if la.get(q.id) == "faithful") / len(gold.meaning_questions))
                mb.append(sum(1 for q in gold.meaning_questions if lb.get(q.id) == "faithful") / len(gold.meaning_questions))
                record.append({"consumer": consumer, "arm": arm, "trial": t,
                               "answer": [a.model_dump() for a in answer.meaning],
                               "verdict_a": la, "verdict_b": lb})
                agree = sum(1 for q in gold.meaning_questions if la.get(q.id) == lb.get(q.id))
                print(f"  [{idx}/{total}] {arm:9} {consumer:12} t{t}  agree {agree}/{len(gold.meaning_questions)}",
                      file=sys.stderr)
    fh.close()
    (out.with_suffix(".json")).write_text(json.dumps(record, indent=2))

    po = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")
    kappa = _cohens_kappa(pairs)
    mean_a = sum(ma) / len(ma) if ma else float("nan")
    mean_b = sum(mb) / len(mb) if mb else float("nan")
    print(f"\nwrote {out.name} ({len(pairs)} item-pairs)")
    print(f"  per-item agreement : {po:.3f}")
    print(f"  Cohen's kappa      : {kappa:.3f}")
    print(f"  meaning_score  judge A ({args.judge_a}) = {mean_a:.3f}   "
          f"judge B ({args.judge_b}) = {mean_b:.3f}   Δ = {mean_b - mean_a:+.3f}")
    print("  Reading: high agreement + kappa and a near-zero meaning_score gap => the metric is not a "
          "house-judge artefact (frozen answers, so this is pure judge variance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
