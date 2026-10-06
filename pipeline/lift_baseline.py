#!/usr/bin/env python3
"""The pre-lift lexical baseline: what does the lift buy over lexical matching?

The reference-anatomy study showed the agent reconciles *lifted semantic models*, not
lexica: the planted false cognates are defeated on the lift's non-lexical content (class,
structural relations, instances), not on labels. This driver measures that directly. It
masks each concept back to its lexical surface (label + synonyms, always present) and adds
the lift's knowledge one factor at a time:

    explanation = the concept's own gloss + worked example
    class       = the shallow kind
    structure   = the structural relations to other concepts
    instances   = the concrete instances (A-box data)

The full 2^4 = 16 subsets give each layer's main effect and the interactions. Two readings
fall out: the cumulative ladder (lexical-only -> +explanation -> +class -> +structure ->
+instances) showing where the trap is defeated, and the lexical-match slice (lexical-only vs
lexical+explanation) that mirrors a with/without-explanatory-support evaluation. The
expected ordering: lexical-only < lexical + explanation < full lifted model.

Run at both-cognitive (the clean lexical-match analogue; content-masking and inertness are
separate axes). Reference held off by default (isolating the lift's own contribution);
--reference turns on the full reference as a second arm. Correctness is the currency;
reasoning tokens recorded for description. Writes results/lift_baseline_<case>[_ref].csv,
row by row (en-route capture), and resumes by default.

    python lift_baseline.py --trials 6
    python lift_baseline.py --trials 6 --reference          # reference-on arm
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Case          # noqa: E402
from reconcile.metrics import score        # noqa: E402
from run import load_dotenv                # noqa: E402

# lift factors added on top of the always-present lexical base (label + synonyms)
FACTORS = ["explanation", "class", "structure", "instances"]

COLUMNS = ["case", "model", "placement", "cell", "uses_reference",
           "explanation", "class", "structure", "instances", "n_fields",
           "trial", "precision", "recall", "surviving_false_cognates", "residual",
           "reasoning_tokens", "surviving_traps"]


def cell_label(flags: dict) -> str:
    on = [f for f in FACTORS if flags[f]]
    return "+".join(on) if on else "lexical-only"


def concept_fields_for(flags: dict) -> set:
    return {f for f in FACTORS if flags[f]}


def trap_names(gold):
    return {frozenset((fc["a"], fc["b"])): f"{fc['a']}~{fc['b']}" for fc in gold.false_cognates}


def surviving_traps(rec, gold, names) -> str:
    hit = set(rec.proposed) & gold.false_cognate_pairs
    return ";".join(sorted(names.get(p, "?") for p in hit))


def _row(case, model, placement, flags, use_ref, trial, rec, gold, names):
    m = score(rec, gold)
    return {
        "case": case, "model": str(model), "placement": placement,
        "cell": cell_label(flags), "uses_reference": str(use_ref),
        "explanation": int(flags["explanation"]), "class": int(flags["class"]),
        "structure": int(flags["structure"]), "instances": int(flags["instances"]),
        "n_fields": sum(int(flags[f]) for f in FACTORS), "trial": trial,
        "precision": m["precision"], "recall": m["recall"],
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "surviving_traps": surviving_traps(rec, gold, names),
    }


def _run_key(r):
    return (str(r["model"]), r["placement"], r["cell"], str(r["uses_reference"]), str(r["trial"]))


def conditions(placements):
    for placement in placements:
        for combo in itertools.product([False, True], repeat=len(FACTORS)):
            yield placement, dict(zip(FACTORS, combo))


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    for model in models:
        print(f"\n{'#'*10} model: {model} {'#'*10}")
        _summarize_one([r for r in rows if r["model"] == model])


def _summarize_one(rows):
    def mean(rs, k):
        xs = [float(x[k]) for x in rs if x[k] not in ("", None)]
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== The lift ladder (cumulative; mean over trials) ===")
    ladder = [("lexical-only", set()),
              ("+explanation", {"explanation"}),
              ("+class", {"explanation", "class"}),
              ("+structure", {"explanation", "class", "structure"}),
              ("+instances (full)", {"explanation", "class", "structure", "instances"})]
    for label, on in ladder:
        sel = [r for r in rows if {f for f in FACTORS if r[f] == 1} == on]
        if sel:
            print(f"    {label.ljust(20)} precision {mean(sel,'precision'):.2f}  "
                  f"recall {mean(sel,'recall'):.2f}  surv.fc {mean(sel,'surviving_false_cognates'):.2f}")

    print("\n=== Main effect of each lift factor (2^4 cells) ===")
    print("    " + "factor".ljust(12) + "surv.false-cog (0->1)      precision (0->1)")
    for f in FACTORS:
        on = [r for r in rows if r[f] == 1]
        off = [r for r in rows if r[f] == 0]
        print("    " + f.ljust(12)
              + f"{mean(off,'surviving_false_cognates'):.2f} -> {mean(on,'surviving_false_cognates'):.2f}".ljust(27)
              + f"{mean(off,'precision'):.2f} -> {mean(on,'precision'):.2f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_big_hard")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano",
                    help="comma-separated model list")
    ap.add_argument("--placements", default="both_cognitive",
                    help="content-masking is studied at both_cognitive; inertness is a "
                         "separate axis")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--reference", action="store_true",
                    help="turn the full shared reference on (second arm); off by default so "
                         "the lift's own contribution is isolated")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results file and start over (default: resume)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack, REF_FIELDS_ALL

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    names = trap_names(case.gold)
    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = max(1, args.trials)
    use_ref = bool(args.reference)
    ref_fields = REF_FIELDS_ALL if use_ref else None

    plan = list(conditions(placements))
    total = len(plan) * trials * len(models)

    out = ROOT / "results" / f"lift_baseline_{args.case}{'_ref' if use_ref else ''}.csv"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {_run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    print(f"Pre-lift baseline (reference {'on' if use_ref else 'off'}): "
          f"{len(models)} model(s) x {len(plan)} content cells x {trials} trials = {total} runs, "
          f"case {args.case}", file=sys.stderr)

    fh = writer = None
    if write:
        fresh_file = args.fresh or not out.exists()
        fh = out.open("w" if fresh_file else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh_file:
            writer.writeheader()
            fh.flush()
            rows, done = [], set()

    i = 0
    for model in models:
        for placement, flags in plan:
            cfields = concept_fields_for(flags)
            label = cell_label(flags)
            stack = None
            for t in range(trials):
                i += 1
                key = (str(model), placement, label, str(use_ref), str(t))
                if key in done:
                    print(f"[{i}/{total}] {model} / {placement} / {label} / trial {t+1}  (captured, skip)",
                          file=sys.stderr, flush=True)
                    continue
                print(f"[{i}/{total}] {model} / {placement} / {label} / trial {t+1}",
                      file=sys.stderr, flush=True)
                if stack is None:
                    stack = OpenAIAgentStack(use_reference=use_ref, model=model,
                                             ref_fields=ref_fields, concept_fields=cfields)
                try:
                    rec = stack.reconcile(case.model_a, case.model_b,
                                          reference=case.reference, placement=placement)
                except Exception as e:
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                r = _row(args.case, model, placement, flags, use_ref, t, rec, case.gold, names)
                rows.append(r)
                done.add(key)
                if writer is not None:
                    writer.writerow(r)
                    fh.flush()
                print(f"      prec {r['precision']} rec {r['recall']} "
                      f"surv.fc {r['surviving_false_cognates']} traps[{r['surviving_traps']}]",
                      file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
