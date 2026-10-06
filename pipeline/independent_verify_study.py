#!/usr/bin/env python3
"""Independent verification vs the same-model self-check (Brad's point 6).

The headline settings verify with the SAME model that produced the reconciliation
(`reconcile_and_verify` calls `verify()` on the same stack). That is a self-check, not an
independent ratifier. This study reconciles once with a fixed model, then verifies the SAME
proposed correspondences with several verifier models -- the reconciler itself (the self-check
baseline) and one or more DIFFERENT models (independent verifiers). Because `verify()` re-derives
from the two models and the proposed pairs, and holds no reasoning from the reconcile step, a
different-model verifier is genuinely independent.

What to read: does an independent verifier keep the same correspondences the self-check keeps
(so the self-check was not inflating results), or does it drop or rescue some (so who verifies
matters)? Precision and resolved fraction are scored before verification (pre) and after each
verifier (post).

  # cheap: reconcile with sol; verify with sol (self) and mini (independent):
  python pipeline/independent_verify_study.py

  # add nano as a third, weaker verifier, and both reference conditions:
  python pipeline/independent_verify_study.py --verifiers gpt-5.6-sol,gpt-5-mini,gpt-5-nano

Resumable. Needs OPENAI_API_KEY. Note: the *executed* Oracle check (verify_modes.virtual_operation)
is instance-level and is demonstrated in the verify sub-study; it is not re-authored here.
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
from reconcile.metrics import score                                    # noqa: E402
from reconcile.stacks.base import Reconciliation                       # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402

CASES = [
    ("config_tapi_teas", "1 · configuration (flagship)"),
    ("config_big_hard", "1 · configuration (hard)"),
    ("config_cross_domain", "3 · cross-domain"),
    ("config_observability", "4 · observability"),
]
COLS = ["case", "setting", "reconciler", "verifier", "self_check", "uses_reference", "trial",
        "precision_pre", "resolved_pre", "precision_post", "resolved_post",
        "surviving_fc_pre", "surviving_fc_post", "proposed", "kept", "dropped",
        "verify_reasoning_tokens"]


def _score_pairs(pairs, a, b, gold, use_ref):
    a_ids, b_ids = set(a.by_id), set(b.by_id)
    matched_a = {next(i for i in p if i in a_ids) for p in pairs}
    matched_b = {next(i for i in p if i in b_ids) for p in pairs}
    rec = Reconciliation(
        stack="verify-study", uses_reference=use_ref, placement="both_cognitive",
        proposed=list(pairs),
        residual_a=[c.id for c in a.concepts if c.id not in matched_a],
        residual_b=[c.id for c in b.concepts if c.id not in matched_b],
    )
    return score(rec, gold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconciler", default="gpt-5.6-sol")
    ap.add_argument("--verifiers", default="gpt-5.6-sol,gpt-5-mini",
                    help="comma-separated verifier models; the one equal to --reconciler is the self-check")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--ref", choices=["0", "1", "both"], default="both")
    ap.add_argument("--out", default=str(ROOT / "results" / "independent_verify_study.csv"))
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    verifiers = [m.strip() for m in args.verifiers.split(",") if m.strip()]
    ref_values = {"0": (False,), "1": (True,), "both": (False, True)}[args.ref]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add((r["case"], r["reconciler"], r["verifier"], r["uses_reference"], r["trial"]))
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    n = 0
    for case_name, setting in CASES:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        for use_ref in ref_values:
            for t in range(args.trials):
                # reconcile once per (case, ref, trial), then verify with each verifier model
                if all((case_name, args.reconciler, vm, str(use_ref), str(t)) in seen for vm in verifiers):
                    continue
                reconciler = OpenAIAgentStack(use_reference=use_ref, model=args.reconciler)
                reference = case.reference if use_ref else None
                try:
                    pre = reconciler.reconcile(case.model_a, case.model_b,
                                               reference=reference, placement="both_cognitive")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! reconcile skip {case_name}/ref={use_ref}/t{t}: {e}", file=sys.stderr)
                    continue
                s_pre = _score_pairs(pre.proposed, case.model_a, case.model_b, case.gold, use_ref)
                for vm in verifiers:
                    key = (case_name, args.reconciler, vm, str(use_ref), str(t))
                    if key in seen:
                        continue
                    verifier = OpenAIAgentStack(use_reference=use_ref, model=vm)
                    try:
                        kept, veff = verifier.verify(case.model_a, case.model_b, reference,
                                                     "both_cognitive", pre.proposed)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! verify skip {case_name}/{vm}/ref={use_ref}/t{t}: {e}", file=sys.stderr)
                        continue
                    s_post = _score_pairs(kept, case.model_a, case.model_b, case.gold, use_ref)
                    row = {
                        "case": case_name, "setting": setting, "reconciler": args.reconciler,
                        "verifier": vm, "self_check": (vm == args.reconciler),
                        "uses_reference": use_ref, "trial": t,
                        "precision_pre": s_pre.get("precision", ""), "resolved_pre": s_pre.get("recall", ""),
                        "precision_post": s_post.get("precision", ""), "resolved_post": s_post.get("recall", ""),
                        "surviving_fc_pre": s_pre.get("surviving_false_cognates", ""),
                        "surviving_fc_post": s_post.get("surviving_false_cognates", ""),
                        "proposed": len(pre.proposed), "kept": len(kept),
                        "dropped": len(pre.proposed) - len(kept),
                        "verify_reasoning_tokens": veff.get("verify_reasoning_tokens", ""),
                    }
                    w.writerow(row); fh.flush(); n += 1
                    tag = "self" if row["self_check"] else "indep"
                    print(f"  {case_name:22} recon={args.reconciler:12} verify={vm:12} ({tag}) "
                          f"ref={int(use_ref)}  P {row['precision_pre']}->{row['precision_post']} "
                          f"RF {row['resolved_pre']}->{row['resolved_post']} dropped={row['dropped']}",
                          file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Read: does the independent verifier (self_check=False) keep the same correspondences as "
          "the self-check (self_check=True)? If yes, the self-check was not inflating; if it drops or "
          "rescues some, who verifies matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
