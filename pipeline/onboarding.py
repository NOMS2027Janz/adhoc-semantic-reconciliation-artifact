#!/usr/bin/env python3
"""E2 -- onboarding cost, reuse vs pairwise: T6 made empirical (Brad review, point 5).

T6 says a shared reference makes reconciliation work grow LINEARLY in the number of systems
against QUADRATICALLY without one. In the study that is a structural count (scaling_otn), not an
agent result, and reuse -- onboarding a genuinely new system against an established reference --
was never exercised. This driver exercises it.

Given N systems over one domain, each already lifted and (for the reuse arm) bound to a shared
reference, we onboard them one at a time, S1 as the seed:

  * REUSE (reference-mediated): onboard S_k with ONE agent pass -- reconcile S_k against the
    seed S1 with the shared reference present. S_k's correspondences to every earlier system
    S_j (j<k) are then DERIVED for free through the shared reference (S_k~S1 from the agent,
    S1~S_j known by shared binding => S_k~S_j). Agent passes to onboard N systems: N-1 (linear).

  * PAIRWISE (no shared reference): onboard S_k by reconciling it directly against each earlier
    system, no reference. Agent passes to onboard N systems: 1+2+...+(N-1) = N(N-1)/2 (quadratic).

Both arms are scored against the same gold (derived by shared reference entry), so we see not
only the cost gap but whether reuse keeps the correspondences correct. Controls (no API) use the
ReferenceReconciler for the reuse derivation and the BaselineMatcher for pairwise, to validate
the counting and the gold offline.

Usage:
  python pipeline/onboarding.py --case scaling_otn --no-write                 # controls only
  python pipeline/onboarding.py --case scaling_otn --agent \\
      --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
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

from reconcile.model import SemanticModel                     # noqa: E402
from reconcile.reference import Reference                     # noqa: E402
from reconcile.stacks.base import Reconciliation              # noqa: E402
from reconcile.stacks.baseline_matcher import BaselineMatcher # noqa: E402
from reconcile.stacks.reference_matcher import ReferenceReconciler  # noqa: E402

SYS_ORDER = ["a", "b", "c", "d", "e", "f", "g", "h"]
COLS = ["case", "arm", "model", "trial", "n_systems", "agent_passes", "total_tokens",
        "precision", "recall", "f1", "false_pos"]


def load_systems(case_dir: Path):
    """Return [(name, SemanticModel)] in a stable order, plus the shared Reference."""
    systems = []
    for s in SYS_ORDER:
        p = case_dir / f"model_{s}.json"
        if p.exists():
            systems.append((s, SemanticModel.from_json(p)))
    ref = Reference.from_json(case_dir / "reference.json")
    return systems, ref


def ref_of(model: SemanticModel) -> dict:
    return {c.id: c.ref for c in model.concepts if c.ref}


def gold_pairs(mi: SemanticModel, mj: SemanticModel) -> set:
    """True correspondences between two systems: concepts sharing a reference entry."""
    bi, bj = ref_of(mi), ref_of(mj)
    by_ref_j = {}
    for cid, r in bj.items():
        by_ref_j.setdefault(r, []).append(cid)
    pairs = set()
    for cid, r in bi.items():
        for jd in by_ref_j.get(r, []):
            pairs.add(frozenset((cid, jd)))
    return pairs


def score_pairs(proposed: set, gold: set) -> dict:
    tp = len(proposed & gold); fp = len(proposed - gold); fn = len(gold - proposed)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "false_pos": fp}


def induce_via_reference(proposed_k1: set, seed: SemanticModel, target: SemanticModel) -> set:
    """From S_k~S1 (agent) and the shared reference, induce S_k~S_target.
    seed==S1. For each (k_concept, s1_concept) the seed's ref fixes an entry; the target's
    concept on that entry is the induced partner."""
    seed_ref = ref_of(seed)
    tgt_by_ref = {}
    for cid, r in ref_of(target).items():
        tgt_by_ref.setdefault(r, []).append(cid)
    induced = set()
    for pair in proposed_k1:
        ids = list(pair)
        # identify which id is the seed's
        s1 = next((i for i in ids if i in seed_ref), None)
        k = next((i for i in ids if i != s1), None)
        if s1 is None or k is None:
            continue
        r = seed_ref[s1]
        for t in tgt_by_ref.get(r, []):
            induced.add(frozenset((k, t)))
    return induced


def reconcile(stack, a, b, ref):
    """Uniform call returning (Reconciliation, tokens)."""
    rec = stack.reconcile(a, b, reference=ref, placement="both_cognitive")
    return rec, int((rec.effort or {}).get("total_tokens") or 0)


def run_arm_reuse(systems, ref, make_agent):
    """Onboard S2..SN, each by ONE pass against the seed S1 with the reference; derive the rest.
    Returns (agent_passes, total_tokens, [per-pair score dicts])."""
    (s1_name, S1) = systems[0]
    passes = 0; tokens = 0; scores = []
    for k in range(1, len(systems)):
        name_k, Sk = systems[k]
        stack = make_agent(use_reference=True)
        rec, tk = reconcile(stack, Sk, S1, ref)
        passes += 1; tokens += tk
        prop_k1 = set(rec.proposed)
        scores.append(score_pairs(prop_k1, gold_pairs(Sk, S1)))       # the direct onboarding pass
        for j in range(1, k):                                          # derived, no extra pass
            name_j, Sj = systems[j]
            induced = induce_via_reference(prop_k1, S1, Sj)
            scores.append(score_pairs(induced, gold_pairs(Sk, Sj)))
    return passes, tokens, scores


def run_arm_pairwise(systems, make_agent):
    """Onboard S2..SN, each by direct reconciliation against every earlier system, no reference."""
    passes = 0; tokens = 0; scores = []
    for k in range(1, len(systems)):
        name_k, Sk = systems[k]
        for j in range(0, k):
            name_j, Sj = systems[j]
            stack = make_agent(use_reference=False)
            rec, tk = reconcile(stack, Sk, Sj, None)
            passes += 1; tokens += tk
            scores.append(score_pairs(set(rec.proposed), gold_pairs(Sk, Sj)))
    return passes, tokens, scores


def mean(xs): return round(sum(xs) / len(xs), 3) if xs else 0.0


def summarise(scores):
    return {"precision": mean([s["precision"] for s in scores]),
            "recall": mean([s["recall"] for s in scores]),
            "f1": mean([s["f1"] for s in scores]),
            "false_pos": sum(s["false_pos"] for s in scores)}


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 onboarding cost: reuse vs pairwise (T6 empirical).")
    ap.add_argument("--case", default="scaling_otn")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    case_dir = ROOT / "benchmark" / "cases" / args.case
    systems, ref = load_systems(case_dir)
    N = len(systems)
    print(f"case {args.case}: {N} systems ({', '.join(n for n, _ in systems)}), "
          f"shared reference {len(ref.entries)} entries")
    print(f"prediction: reuse onboards in N-1 = {N-1} passes (linear); "
          f"pairwise in N(N-1)/2 = {N*(N-1)//2} passes (quadratic)\n")

    rows = []

    # ---- controls (no API): validate the counting and the gold ----
    def ctrl_ref(use_reference=True): return ReferenceReconciler()
    def ctrl_base(use_reference=False): return BaselineMatcher()
    p, tk, sc = run_arm_reuse(systems, ref, ctrl_ref)
    rows.append({"case": args.case, "arm": "reuse (control: reference)", "model": "-", "trial": 0,
                 "n_systems": N, "agent_passes": p, "total_tokens": tk, **summarise(sc)})
    p, tk, sc = run_arm_pairwise(systems, ctrl_base)
    rows.append({"case": args.case, "arm": "pairwise (control: label)", "model": "-", "trial": 0,
                 "n_systems": N, "agent_passes": p, "total_tokens": tk, **summarise(sc)})

    # ---- agent arms ----
    if args.agent:
        from run import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
            return 2
        from reconcile.stacks.agent_openai import OpenAIAgentStack
        for model in [m.strip() for m in args.model.split(",") if m.strip()]:
            def make(use_reference, _m=model):
                return OpenAIAgentStack(use_reference=use_reference, model=_m)
            for t in range(max(1, args.trials)):
                try:
                    p, tk, sc = run_arm_reuse(systems, ref, make)
                    rows.append({"case": args.case, "arm": "reuse (agent)", "model": model, "trial": t,
                                 "n_systems": N, "agent_passes": p, "total_tokens": tk, **summarise(sc)})
                    print(f"  reuse    {model} t{t}: {p} passes, {tk} tok, "
                          f"prec {rows[-1]['precision']} recall {rows[-1]['recall']}", file=sys.stderr)
                    p, tk, sc = run_arm_pairwise(systems, make)
                    rows.append({"case": args.case, "arm": "pairwise (agent)", "model": model, "trial": t,
                                 "n_systems": N, "agent_passes": p, "total_tokens": tk, **summarise(sc)})
                    print(f"  pairwise {model} t{t}: {p} passes, {tk} tok, "
                          f"prec {rows[-1]['precision']} recall {rows[-1]['recall']}", file=sys.stderr)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {model} t{t}: {e}", file=sys.stderr)

    # ---- report ----
    w = max(len(r["arm"]) for r in rows)
    print(f"\n{'arm':{w}} {'model':13}{'passes':7}{'tokens':8}{'prec':6}{'recall':7}{'fp':4}")
    for r in rows:
        print(f"{r['arm']:{w}} {r['model']:13}{r['agent_passes']:<7}{r['total_tokens']:<8}"
              f"{r['precision']:<6}{r['recall']:<7}{r['false_pos']:<4}")

    if not args.no_write:
        out = Path(args.out) if args.out else ROOT / "results" / f"onboarding_{args.case}.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=COLS); wr.writeheader()
            for r in rows:
                wr.writerow(r)
        print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
