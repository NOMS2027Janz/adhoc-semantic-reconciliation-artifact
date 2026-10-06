#!/usr/bin/env python3
"""The lift study: is the lift agent-performable, and is reconciliation invariant to who performed it?

Brad's point 1 asks whether the study tests *who performs the lift* or only reconciliation over an
already-lifted model. The benchmark's lifted models are pre-materialised fixtures standing for what a
cognitive side produces when it lifts and explains itself. This runner instruments that step: for each
case it runs, at both_cognitive, two arms and scores both against the same validated gold.

  * fixture   -> reconcile the fixture models as-is (the explanation layer is the authored fixture)
  * agent-lift -> an agent first LIFTS each side (produces gloss + example from that side's schema
                  surface alone; see src/reconcile/lift.py), then reconcile the agent-lifted models

Both arms reconcile with the two-agent stack (the honest both-cognitive), so no single-agent
both-cognitive numbers are created. The lift is computed once per (case, model, trial) and reused across
the reference conditions. Quality: precision, resolved fraction, surviving false cognates, residual.
Lift: reasoning tokens, coverage (fraction of concepts lifted), and mean gloss fidelity vs the fixture.

  # cheap first pass (strong model, one trial, the four core cases, both ref conditions):
  python pipeline/lift_study.py

  # fuller (a few rungs, more trials):
  python pipeline/lift_study.py --model gpt-5.6-sol,gpt-5-mini --trials 2

  # a single case while iterating:
  python pipeline/lift_study.py --cases config_tapi_teas --ref 0

Resumable: existing rows in results/lift_study.csv are skipped. Needs OPENAI_API_KEY (Chris runs it).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

from reconcile.model import Case                                        # noqa: E402
from reconcile.metrics import score                                    # noqa: E402
from reconcile.lift import lift_model, gloss_fidelity                  # noqa: E402
from reconcile.stacks.agent_openai import OpenAIAgentStack             # noqa: E402
from reconcile.stacks.agent_twoagent import TwoAgentStack             # noqa: E402
from run import load_dotenv                                            # noqa: E402
from two_agent_study import SETTING_OF                                 # noqa: E402

CASES = [
    ("config_tapi_teas", "1 · configuration (flagship)"),
    ("config_big_hard", "1 · configuration (hard)"),
    ("config_cross_domain", "3 · cross-domain"),
    ("config_observability", "4 · observability"),
]
COLS = ["case", "setting", "arm", "model", "uses_reference", "trial",
        "precision", "resolved_fraction", "surviving_false_cognates", "residual",
        "turns", "total_tokens", "reasoning_tokens", "latency_s",
        "lift_reasoning_tokens", "lift_coverage", "gloss_fidelity"]


def _make_stack(stack_kind, use_ref, model, max_rounds):
    if stack_kind == "single":
        return OpenAIAgentStack(use_reference=use_ref, model=model)
    return TwoAgentStack(use_reference=use_ref, model=model, max_rounds=max_rounds)


def _dump_lifted(sm, dump_dir: Path, case_name: str, side: str, model: str, trial: int,
                 evidence: dict | None = None) -> Path:
    """Write one agent-produced lifted model to JSON in the fixture's own shape, so it can be
    diffed concept-for-concept against benchmark/cases/<case>/<side>*.json. When `evidence` is given
    (--trace), each concept also carries the agent's one-line stated reasoning, for a 'lift in the act'."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    note = (f"Agent-produced lift: the gloss and example of each concept were written by {model} from "
            f"that side's schema surface alone (label, synonyms, kind, relations, and instances); the "
            f"reference binding (ref) and every other field are carried from the source unchanged. This "
            f"is the {side} side of case {case_name}, trial {trial}. Compare concept-for-concept against "
            f"the fixture at benchmark/cases/{case_name}/{side}*.json.")
    d = sm.to_dict(note=note)
    if evidence:
        for c in d["concepts"]:
            ev = evidence.get(c["id"])
            if ev:
                c["evidence"] = ev
    path = dump_dir / f"{case_name}__{side}__{model}__t{trial}.json"
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return path


def _mean_fidelity(agent_a, fixture_a, agent_b, fixture_b) -> float:
    fx = {c.id: c.gloss for c in fixture_a.concepts} | {c.id: c.gloss for c in fixture_b.concepts}
    vals = [gloss_fidelity(c.gloss, fx.get(c.id, ""))
            for c in list(agent_a.concepts) + list(agent_b.concepts)]
    return round(mean(vals), 3) if vals else ""


def _row(case_name, setting, arm, model, use_ref, trial, rec, scored, lift_eff):
    return {
        "case": case_name, "setting": setting, "arm": arm, "model": model,
        "uses_reference": use_ref, "trial": trial,
        "precision": scored.get("precision", ""),
        "resolved_fraction": scored.get("recall", ""),   # internal scorer key is 'recall'
        "surviving_false_cognates": scored.get("surviving_false_cognates", ""),
        "residual": scored.get("residual", ""),
        "turns": rec.work.get("turns", ""),
        "total_tokens": rec.effort.get("total_tokens", ""),
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "latency_s": rec.effort.get("latency_s", ""),
        "lift_reasoning_tokens": lift_eff.get("lift_reasoning_tokens", "") if lift_eff else "",
        "lift_coverage": lift_eff.get("lift_coverage", "") if lift_eff else "",
        "gloss_fidelity": lift_eff.get("gloss_fidelity", "") if lift_eff else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=6, help="two-agent negotiation round cap")
    ap.add_argument("--ref", choices=["0", "1", "both"], default="both")
    ap.add_argument("--stack", choices=["single", "two-agent"], default="two-agent",
                    help="reconciliation architecture for BOTH arms (default two-agent, the honest "
                         "both-cognitive; single is a cheaper smoke that only varies the lift source)")
    ap.add_argument("--cases", default="",
                    help="comma-separated case names instead of the four core schema cases, or 'all' to "
                         "run every schema-matching case under benchmark/cases/ (cases not shaped for the "
                         "lift study, e.g. intent_*/scaling_*, are skipped)")
    ap.add_argument("--out", default=str(ROOT / "results" / "lift_study.csv"))
    ap.add_argument("--dump-lifted-dir", default=str(ROOT / "results" / "agent_lifted_models"),
                    help="write each agent-produced lifted model to this directory as JSON, in the "
                         "fixture's own shape, for concept-for-concept comparison against the fixtures; "
                         "pass an empty string to disable")
    ap.add_argument("--lift-only", action="store_true",
                    help="just perform and write each agent lift (no reconciliation, no CSV): the "
                         "cheapest way to inspect the agent-produced models. Ignores CSV resume state.")
    ap.add_argument("--trace", action="store_true",
                    help="with --lift-only, also record the agent's one-line stated EVIDENCE per concept "
                         "(the surface signals it used and the look-alike it ruled out) into each dumped "
                         "model, so the lift can be shown 'in the act'. Display only; not fed to reconciliation.")
    args = ap.parse_args()

    ref_values = {"0": (False,), "1": (True,), "both": (False, True)}[args.ref]
    if not args.cases:
        cases = CASES
    elif args.cases.strip().lower() == "all":
        cdir = ROOT / "benchmark" / "cases"
        cases = [(p.name, SETTING_OF.get(p.name, p.name))
                 for p in sorted(cdir.iterdir())
                 if p.is_dir() and (p / "gold.json").exists()]
    else:
        cases = [(c.strip(), SETTING_OF.get(c.strip(), c.strip()))
                 for c in args.cases.split(",") if c.strip()]

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.model.split(",") if m.strip()]

    if args.lift_only:
        dd = Path(args.dump_lifted_dir or (ROOT / "results" / "agent_lifted_models"))
        made = 0
        for case_name, setting in cases:
            try:
                case = Case.load(ROOT / "benchmark" / "cases" / case_name)
            except Exception as e:  # noqa: BLE001 - skip cases not shaped for the lift study
                print(f"  - skip {case_name}: not a schema-matching case ({e})", file=sys.stderr)
                continue
            for model in models:
                for t in range(args.trials):
                    try:
                        la, ea = lift_model(case.model_a, model, trace=args.trace)
                        lb, eb = lift_model(case.model_b, model, trace=args.trace)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! lift failed {case_name}/{model}/t{t}: {e}", file=sys.stderr)
                        continue
                    pa = _dump_lifted(la, dd, case_name, "model_a", model, t, ea.get("trace"))
                    pb = _dump_lifted(lb, dd, case_name, "model_b", model, t, eb.get("trace"))
                    cov = round(mean([ea.get("lift_coverage", 0), eb.get("lift_coverage", 0)]), 3)
                    fid = _mean_fidelity(la, case.model_a, lb, case.model_b)
                    made += 2
                    print(f"  {case_name:22} {model:12} t{t}  cov={cov} fid={fid}  "
                          f"-> {pa.name}, {pb.name}", file=sys.stderr)
        print(f"\nwrote {made} agent-lifted model file(s) to {dd.name}/.")
        print("Each is the agent's lift in the fixture's own shape; diff it against the matching "
              "benchmark/cases/<case>/<side>*.json to read the produced gloss/example against the fixture.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    seen = set()
    if out.exists():
        with out.open() as fh:
            for r in csv.DictReader(fh):
                seen.add((r["case"], r["arm"], r["model"], r["uses_reference"], r["trial"]))
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    n = 0
    for case_name, setting in cases:
        try:
            case = Case.load(ROOT / "benchmark" / "cases" / case_name)
        except Exception as e:  # noqa: BLE001 - skip cases not shaped for the lift study
            print(f"  - skip {case_name}: not a schema-matching case ({e})", file=sys.stderr)
            continue
        for model in models:
            for t in range(args.trials):
                # the lift is reference-independent: compute it once per (case, model, trial) if any
                # agent-lift row for this cell is still outstanding, then reuse across ref conditions.
                lifted_a = lifted_b = None
                lift_eff = None
                need_lift = any(
                    (case_name, "agent-lift", model, str(ur), str(t)) not in seen
                    for ur in ref_values
                )
                if need_lift:
                    try:
                        lifted_a, eff_a = lift_model(case.model_a, model)
                        lifted_b, eff_b = lift_model(case.model_b, model)
                        lift_eff = {
                            "lift_reasoning_tokens":
                                (eff_a.get("lift_reasoning_tokens") or 0)
                                + (eff_b.get("lift_reasoning_tokens") or 0),
                            "lift_coverage": round(
                                mean([eff_a.get("lift_coverage", 0), eff_b.get("lift_coverage", 0)]), 3),
                            "gloss_fidelity": _mean_fidelity(lifted_a, case.model_a,
                                                             lifted_b, case.model_b),
                        }
                        if args.dump_lifted_dir:
                            dd = Path(args.dump_lifted_dir)
                            pa = _dump_lifted(lifted_a, dd, case_name, "model_a", model, t)
                            pb = _dump_lifted(lifted_b, dd, case_name, "model_b", model, t)
                            print(f"    dumped agent lift -> {pa.name}, {pb.name}", file=sys.stderr)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! lift failed {case_name}/{model}/t{t}: {e}", file=sys.stderr)
                        lifted_a = lifted_b = None

                for use_ref in ref_values:
                    arms = [("fixture", case.model_a, case.model_b, None)]
                    if lifted_a is not None and lifted_b is not None:
                        arms.append(("agent-lift", lifted_a, lifted_b, lift_eff))
                    for arm, ma, mb, eff in arms:
                        key = (case_name, arm, model, str(use_ref), str(t))
                        if key in seen:
                            continue
                        try:
                            stack = _make_stack(args.stack, use_ref, model, args.max_rounds)
                            reference = case.reference if stack.uses_reference else None
                            rec = stack.reconcile(ma, mb, reference=reference,
                                                  placement="both_cognitive")
                            scored = score(rec, case.gold)
                        except Exception as e:  # noqa: BLE001 - one bad run must not sink the job
                            print(f"  ! skip {case_name}/{arm}/{model}/ref={use_ref}/t{t}: {e}",
                                  file=sys.stderr)
                            continue
                        row = _row(case_name, setting, arm, model, use_ref, t, rec, scored, eff)
                        w.writerow(row); fh.flush(); n += 1
                        print(f"  {case_name:22} {arm:10} {model:12} ref={int(use_ref)} t{t}  "
                              f"P={row['precision']} RF={row['resolved_fraction']} "
                              f"sfc={row['surviving_false_cognates']} "
                              f"cov={row['lift_coverage']} fid={row['gloss_fidelity']}",
                              file=sys.stderr)
    fh.close()
    print(f"\nwrote {out.name} — {n} new row(s).")
    if args.dump_lifted_dir:
        print(f"agent-produced lifted models written to {Path(args.dump_lifted_dir).name}/ "
              f"(one JSON per case/side/model/trial, in the fixture's shape). Diff any against its "
              f"fixture, e.g.:  diff <(python -m json.tool benchmark/cases/config_big_hard/model_b_teas.json) "
              f"<(python -m json.tool {Path(args.dump_lifted_dir).name}/config_big_hard__model_b__<model>__t0.json)")
    print("Compare, per case/model/ref: does the agent-lift arm reach the same precision and resolved "
          "fraction as the fixture arm? Invariance (delta approx 0) shows the lift is agent-performable "
          "and reconciliation does not depend on who performed it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
