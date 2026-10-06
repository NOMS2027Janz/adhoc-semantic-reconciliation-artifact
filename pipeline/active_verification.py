#!/usr/bin/env python3
"""E21 -- active verification between two cognitive systems: can peers manufacture the shared
binding authorisation without an authored reference?

The note "Producing the inputs" leaves one question open. The reconciliation study established
that a *reference* supplies the good a hard cross-domain seam turns on -- the shared category that
AUTHORISES a binding and blocks the false cognate ("good (B)") -- and that it does so at every
capability tier, where mere elicitation of meaning ("good (A)") helps only a middle band. But the
study only ever supplied (A) *passively* (E16: a one-shot gloss per concept). It never tested what
two fully cognitive systems can do when they ACTIVELY VERIFY: propose a candidate binding, run a
decisive virtual experiment on it (provision the pair, operate it, read back whether the invariants
a correct translation must preserve hold), and converge on a verified correspondence on EVIDENCE.

E21 tests exactly that, on the hard seams, reference OFF (the interesting regime: can peers make (B)
themselves?), across the capability ladder. Arms, all at both_cognitive:

  * single            -- OpenAIAgentStack, no reference. The deferring single agent (context).
  * two-agent         -- TwoAgentStack, no reference, NO experiment. Negotiation only: propose / ask /
                         answer / ratify under information asymmetry. Supplies (A) by exchange and
                         reasons over it, but has no decisive test. (The L1 comparator.)
  * two-agent+exp     -- TwoAgentStack, no reference, WITH a decisive-experiment oracle (a bounded
                         SchemaOracle). This is active verification: the peers may spend a limited
                         budget of virtual provisionings to settle candidates on evidence. (THE arm.)
  * two-agent/ref     -- TwoAgentStack WITH the reference. Upper bound: (B) handed over, not made.
  * single/ref        -- OpenAIAgentStack WITH the reference. The study's headline upper bound.

Two SYNTHESIS arms are available with --with-frame (off by default), where the peers build their own
shared reference rather than being handed one:

  * two-agent+frame     -- construct a thin shared reference from the two models (construct-then-bind,
                           the negotiating tier's own model, no authored standard) and negotiate
                           through it. Tests: can peers MAKE (B) by construction?
  * two-agent+frame+exp -- the constructed frame PLUS the decisive experiment. Tests: does verification
                           CLOSE the constructed frame's cognate-guard gap and match the reference?

The readout the note needs: does two-agent+exp drive the resolved fraction to one and keep surviving
false cognates at zero WITHOUT a reference, and does it do so at EVERY tier (capability-independent,
like an authored reference) or only at the strong tier (capability-gated, like the observability
probe result E4')? Either answer closes the note's open question. The synthesis arms then ask whether
peers, unaided by any authored standard, reach the reference-given close.

The experiment budget forces choice: it caps virtual provisionings below the A x B pair count, so the
peers must spend experiments on the seams that matter rather than brute-forcing every pair -- knowing
*which* pair to test is itself the skill under test. Default budget = |correct| + |false cognates|.

    # offline: check oracle verdicts against gold + case construction, NO API
    python pipeline/active_verification.py --validate

    # the run (one model per terminal, three terminals = the ladder):
    python pipeline/active_verification.py --models gpt-5.6-sol --trials 3
    python pipeline/active_verification.py --models gpt-5-mini  --trials 3
    python pipeline/active_verification.py --models gpt-5-nano  --trials 3

    # follow-up: the synthesis arms only (base arms already in the CSV are skipped by resume):
    python pipeline/active_verification.py --models gpt-5.6-sol --trials 3 --with-frame

Writes results/active_verification.csv (one row per case x arm x model x ref x trial). Resumable:
existing rows are skipped. Needs OPENAI_API_KEY (except --validate).
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
from reconcile.metrics import score                                     # noqa: E402
from reconcile.schema_oracle import SchemaOracle                        # noqa: E402
from reconcile.construct import construct_reference                     # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack              # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack               # noqa: E402

# the hard seams: cross-domain flagship, the hard configuration case, and the REST reproduction
# (config_rest carries the sharpest cognate -- identical name AND enumerated values).
CASES = ["config_cross_domain", "config_big_hard", "config_rest"]

COLS = ["case", "stack", "model", "uses_reference", "uses_experiment", "exp_budget", "trial",
        "precision", "resolved_fraction", "surviving_false_cognates", "residual",
        "turns", "experiments_used", "total_tokens", "reasoning_tokens", "latency_s"]


def _budget(case: Case, override: int | None) -> int:
    if override is not None:
        return override
    return len(case.gold.correct_pairs) + len(case.gold.false_cognate_pairs)


# arm descriptors: (label, uses_reference, uses_experiment, uses_frame). The stack for each is
# built per-run in main (the +exp arm needs a fresh oracle per trial; the +frame arms construct
# their shared frame with the negotiating tier's own model at run time).
_BASE_ARMS = [
    ("single",        False, False, False),
    ("two-agent",     False, False, False),
    ("two-agent+exp", False, True,  False),
    ("two-agent/ref", True,  False, False),
    ("single/ref",    True,  False, False),
]
# the synthesis arms (opt-in via --with-frame): the peers BUILD their own shared reference from the
# two models (construct-then-bind, no authored standard) and negotiate through it, and +frame+exp
# adds the decisive virtual experiment on top. The question: unaided by any authored standard, do
# peers who construct their own frame -- and can verify it -- match the reference-given close at
# every tier, and does the experiment close the constructed frame's cognate-guard gap?
_FRAME_ARMS = [
    ("two-agent+frame",     False, False, True),
    ("two-agent+frame+exp", False, True,  True),
]
ALL_ARM_NAMES = [a[0] for a in _BASE_ARMS + _FRAME_ARMS]


def _select_arms(with_frame: bool, only: str):
    arms = list(_BASE_ARMS) + (list(_FRAME_ARMS) if with_frame else [])
    if only != "all":
        arms = [a for a in arms if a[0] == only]
    return arms


def _row(case_name, label, model, use_ref, use_exp, budget, trial, rec, scored):
    return {
        "case": case_name, "stack": label, "model": model,
        "uses_reference": use_ref, "uses_experiment": use_exp,
        "exp_budget": budget if use_exp else "", "trial": trial,
        "precision": scored.get("precision", ""),
        "resolved_fraction": scored.get("recall", ""),   # internal metric key is 'recall'
        "surviving_false_cognates": scored.get("surviving_false_cognates", ""),
        "residual": scored.get("residual", ""),
        "turns": rec.work.get("turns", ""),
        "experiments_used": rec.work.get("experiments", ""),
        "total_tokens": rec.effort.get("total_tokens", ""),
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "latency_s": rec.effort.get("latency_s", ""),
    }


def _done_key(r):
    return (r["case"], r["stack"], r["model"], str(r["uses_reference"]), str(r["trial"]))


def validate() -> int:
    """Offline: the decisive-experiment oracle must agree with the gold on every seam, and the
    scoring path must run. No API calls."""
    ok = True
    for case_name in CASES:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        budget = _budget(case, None)
        oracle = SchemaOracle(case, budget=None)   # unbudgeted for the check
        # every true correspondence must read back as identity...
        for pair in case.gold.correct_pairs:
            a_id, b_id = tuple(pair)
            res = oracle.virtual_provision(a_id, b_id)
            if not (res.ok and res.confirmed and res.relation == "identity"):
                print(f"  ! {case_name}: correct pair {set(pair)} -> {res.relation} (confirmed={res.confirmed})")
                ok = False
        # ...and every planted false cognate as refuted/false-cognate
        for pair in case.gold.false_cognate_pairs:
            a_id, b_id = tuple(pair)
            res = oracle.virtual_provision(a_id, b_id)
            if not (res.ok and not res.confirmed and res.relation == "false-cognate"):
                print(f"  ! {case_name}: false cognate {set(pair)} -> {res.relation} (confirmed={res.confirmed})")
                ok = False
        print(f"  {case_name:20} A={len(case.model_a.concepts)} B={len(case.model_b.concepts)} "
              f"correct={len(case.gold.correct_pairs)} false_cog={len(case.gold.false_cognate_pairs)} "
              f"default_budget={budget}  oracle {'OK' if ok else 'FAILED'}")
    print("validate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=6, help="two-agent negotiation round cap")
    ap.add_argument("--exp-budget", type=int, default=None,
                    help="virtual-experiment budget for two-agent+exp (default |correct|+|false cog|)")
    ap.add_argument("--cases", default=",".join(CASES),
                    help="comma-separated case names (default the three hard seams)")
    ap.add_argument("--only-arm", default="all", choices=["all"] + ALL_ARM_NAMES,
                    help="restrict to one arm (e.g. two-agent+exp, or two-agent+frame+exp)")
    ap.add_argument("--with-frame", action="store_true",
                    help="also run the synthesis arms two-agent+frame and two-agent+frame+exp "
                         "(peers construct their own shared reference; off by default)")
    ap.add_argument("--out", default=str(ROOT / "results" / "active_verification.csv"))
    ap.add_argument("--validate", action="store_true",
                    help="offline: check oracle-vs-gold and construction, no API")
    args = ap.parse_args()

    if args.validate:
        return validate()

    # OPENAI_API_KEY via the shared dotenv loader used across the pipeline
    from run import load_dotenv  # noqa: E402
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)

    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add(_done_key(r))
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    arms_list = _select_arms(args.with_frame, args.only_arm)
    total = len(cases) * len(models) * len(arms_list) * args.trials   # for the [idx/total] progress index
    idx = 0
    n = 0
    for case_name in cases:
        case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        budget = _budget(case, args.exp_budget)
        for model in models:
            for label, use_ref, use_exp, use_frame in arms_list:
                is_two_agent = label.startswith("two-agent")
                for t in range(args.trials):
                    idx += 1
                    key = (case_name, label, model, str(use_ref), str(t))
                    if key in seen:
                        continue
                    try:
                        # +frame arms: the peers construct their own shared reference with the
                        # negotiating tier's own model (fresh per trial, for honest variance).
                        frame, con_eff = (construct_reference(case.model_a, case.model_b, model)
                                          if use_frame else (None, {}))
                        # a fresh oracle per run so the budget counter is not shared across trials
                        oracle = SchemaOracle(case, budget=budget) if use_exp else None
                        if is_two_agent:
                            stack = TwoAgentStack(use_reference=use_ref, model=model,
                                                  max_rounds=args.max_rounds,
                                                  oracle=oracle, shared_frame=frame)
                        else:
                            stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                        reference = case.reference if use_ref else None
                        rec = stack.reconcile(case.model_a, case.model_b,
                                              reference=reference, placement="both_cognitive")
                        # fold the frame-construction cost into the arm's totals, so the CSV keeps
                        # one column set and the +frame cost is charged to the arm, not hidden.
                        if con_eff:
                            rec.effort["total_tokens"] = ((rec.effort.get("total_tokens") or 0)
                                                          + (con_eff.get("construct_total_tokens") or 0))
                            rec.effort["reasoning_tokens"] = ((rec.effort.get("reasoning_tokens") or 0)
                                                              + (con_eff.get("construct_reasoning_tokens") or 0))
                        scored = score(rec, case.gold)
                    except Exception as e:  # noqa: BLE001 - one bad run must not sink the job
                        print(f"  [{idx}/{total}] ! skip {case_name}/{label}/{model}/t{t}: {e}", file=sys.stderr)
                        continue
                    row = _row(case_name, label, model, use_ref, use_exp, budget, t, rec, scored)
                    w.writerow(row); fh.flush(); n += 1
                    print(f"  [{idx}/{total}] {case_name:20} {label:14} {model:12} t{t}  "
                          f"P={row['precision']} RF={row['resolved_fraction']} "
                          f"sfc={row['surviving_false_cognates']} exp={row['experiments_used']} "
                          f"turns={row['turns']} tok={row['reasoning_tokens']}", file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    print("Readout: does two-agent+exp reach RF=1 and sfc=0 with NO reference, and at EVERY tier "
          "(capability-independent) or only the strong one (capability-gated)?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
