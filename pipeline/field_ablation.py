#!/usr/bin/env python3
"""Anatomy of the reference: a factorial ablation of its descriptive fields.

The lexical reference entry has an identity anchor (its id) plus five descriptive
fields grouped into four factors:

    lexical    = label + synonyms      (the lexical surface)
    class      = the shallow class     (a coarse type)
    definition = the disambiguating gloss
    example    = one canonical example

The id is the coreference anchor, not descriptive evidence, so it is always present;
the empty cell (no descriptive fields) is the id-only *floor*. We run the full 2^4 =
16 field subsets, which gives every field's main effect AND every interaction (so we
can see, e.g., whether definition and example are substitutes). A separate
no-reference anchor (the reference withheld entirely) sits below the floor.

Importance is measured in the currency that is NOT confounded by prompt length:
CORRECTNESS. Removing a field's information and removing its text are the same act,
so effort (reasoning tokens) cannot cleanly attribute importance to a field; but a
field that carries no disambiguating information cannot change precision, recall,
surviving false cognates, or residual however long its text. We record reasoning
tokens for description only.

The mechanism lives where meaning must be reconstructed, so the core runs are the two
inert placements at the strong model; both-cognitive (binding pre-given) is a small
flatness control. For each run we also record WHICH planted false cognate survived,
so the headline is a fields x traps attribution.

    python field_ablation.py --trials 6                    # core: strong model, both inert
    python field_ablation.py --trials 6 --model gpt-5-nano # capability check (optional)

Writes results/ablation_<case>.csv (one row per run).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Case          # noqa: E402
from reconcile.metrics import score        # noqa: E402
from run import load_dotenv                # noqa: E402

# the four factors, in report order, and the entry fields each turns on
FACTORS = ["lexical", "class", "definition", "example"]
FACTOR_FIELDS = {
    "lexical": ("label", "synonyms"),
    "class": ("class",),
    "definition": ("definition",),
    "example": ("example",),
}

COLUMNS = ["case", "model", "placement", "cell", "uses_reference",
           "lexical", "class", "definition", "example", "n_fields",
           "trial", "precision", "recall", "surviving_false_cognates", "residual",
           "reasoning_tokens", "surviving_traps"]


def cell_label(flags: dict) -> str:
    on = [f for f in FACTORS if flags[f]]
    return "+".join(on) if on else "id-only"


def ref_fields_for(flags: dict) -> set:
    fields: set = set()
    for f in FACTORS:
        if flags[f]:
            fields.update(FACTOR_FIELDS[f])
    return fields


def opaquify(case):
    """Return (reference, model_a, model_b) with every reference id replaced by an
    opaque token (e01, e02, ...) and each concept's binding remapped to match.

    The first pass used human-readable reference ids (link-termination, ...), so the
    id-only floor was not information-free: the identifier named the concept. Opaque
    ids remove that leak, so the descriptive fields become the ONLY evidence for
    binding an inert concept to its entry. Concept ids are untouched (the gold scores
    on them), so scoring is unaffected."""
    ref = case.reference
    idmap = {e.id: f"e{i:02d}" for i, e in enumerate(ref.entries, 1)}
    new_ref = dataclasses.replace(
        ref, entries=[dataclasses.replace(e, id=idmap[e.id]) for e in ref.entries])

    def remap(m):
        return dataclasses.replace(
            m, concepts=[dataclasses.replace(c, ref=idmap.get(c.ref, c.ref))
                         for c in m.concepts])

    return new_ref, remap(case.model_a), remap(case.model_b)


def trap_names(gold):
    """frozenset({a,b}) -> short 'a~b' handle, for per-trap attribution."""
    out = {}
    for fc in gold.false_cognates:
        a, b = fc["a"], fc["b"]
        out[frozenset((a, b))] = f"{a}~{b}"
    return out


def surviving_traps(rec, gold, names) -> str:
    hit = set(rec.proposed) & gold.false_cognate_pairs
    return ";".join(sorted(names.get(p, "?") for p in hit))


def _row(case, model, placement, flags, use_ref, trial, rec, gold, names):
    m = score(rec, gold)
    return {
        "case": case, "model": str(model), "placement": placement,
        "cell": cell_label(flags) if use_ref else "no-reference",
        "uses_reference": str(use_ref),
        "lexical": int(flags["lexical"]), "class": int(flags["class"]),
        "definition": int(flags["definition"]), "example": int(flags["example"]),
        "n_fields": sum(int(flags[f]) for f in FACTORS),
        "trial": trial,
        "precision": m["precision"], "recall": m["recall"],
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "surviving_traps": surviving_traps(rec, gold, names),
    }


def _run_key(r):
    """Identity of one run, for resume/skip: model, placement, cell, ref-flag, trial."""
    return (str(r["model"]), r["placement"], r["cell"], str(r["uses_reference"]), str(r["trial"]))


ALL_ON = {f: True for f in FACTORS}
ALL_OFF = {f: False for f in FACTORS}


def conditions(placements, control):
    """Yield (placement, flags, use_reference) tuples.

    Inert placements get all 16 field subsets (use_reference=True) plus a
    no-reference anchor. The both-cognitive control gets full, id-only, and
    no-reference only (binding is pre-given there, so the fields should not matter)."""
    for placement in placements:
        for combo in itertools.product([False, True], repeat=len(FACTORS)):
            flags = dict(zip(FACTORS, combo))
            yield placement, flags, True
        yield placement, dict(ALL_OFF), False  # no-reference anchor
    if control:
        for flags, use_ref in ((ALL_ON, True), (ALL_OFF, True), (ALL_OFF, False)):
            yield "both_cognitive", dict(flags), use_ref


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    for model in models:
        print(f"\n{'#'*10} model: {model} {'#'*10}")
        _summarize_one([r for r in rows if r["model"] == model])


def _summarize_one(rows):
    inert = [r for r in rows if r["placement"] in ("one_inert", "both_inert")
             and r["uses_reference"] == "True"]
    if not inert:
        return

    def mean(rs, k):
        xs = [float(x[k]) for x in rs if x[k] not in ("", None)]
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== Main effect of each factor (inert placements, 2^4 cells) ===")
    print("    field present vs absent, averaged over all other fields and trials")
    print("    " + "factor".ljust(12) + "surv.false-cog (0->1)      recall (0->1)")
    for f in FACTORS:
        on = [r for r in inert if r[f] == 1]
        off = [r for r in inert if r[f] == 0]
        print("    " + f.ljust(12)
              + f"{mean(off,'surviving_false_cognates'):.2f} -> {mean(on,'surviving_false_cognates'):.2f}".ljust(27)
              + f"{mean(off,'recall'):.2f} -> {mean(on,'recall'):.2f}")

    print("\n=== Per-trap survival by factor (inert placements) ===")
    print("    fraction of runs where the trap survived, field ABSENT vs PRESENT")
    traps = sorted({t for r in inert for t in r["surviving_traps"].split(";") if t})
    for trap in traps:
        print(f"\n  trap {trap}")
        for f in FACTORS:
            on = [r for r in inert if r[f] == 1]
            off = [r for r in inert if r[f] == 0]
            fr_off = sum(1 for r in off if trap in r["surviving_traps"].split(";")) / len(off) if off else 0
            fr_on = sum(1 for r in on if trap in r["surviving_traps"].split(";")) / len(on) if on else 0
            print(f"    {f.ljust(12)} absent {fr_off:.2f}   present {fr_on:.2f}")

    print("\n=== Anchor cells (inert placements, mean over trials) ===")
    for label in ("id-only", "definition+example", "lexical+class+definition+example", "no-reference"):
        sel = [r for r in inert + [x for x in rows if x["uses_reference"] == "False"
               and x["placement"] in ("one_inert", "both_inert")] if r["cell"] == label]
        if sel:
            print(f"    {label.ljust(36)} precision {mean(sel,'precision'):.2f}  "
                  f"recall {mean(sel,'recall'):.2f}  surv.fc {mean(sel,'surviving_false_cognates'):.2f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_big_hard")
    ap.add_argument("--model", default="gpt-5.6-sol",
                    help="comma-separated model list")
    ap.add_argument("--placements", default="both_inert,one_inert")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--opaque-ids", action="store_true",
                    help="replace reference ids with opaque tokens so the descriptive "
                         "fields are the only binding evidence (pass 2)")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the both-cognitive flatness control")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results file and start over "
                         "(default: resume, skipping runs already captured)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack

    case = Case.load(ROOT / "benchmark" / "cases" / args.case)
    names = trap_names(case.gold)
    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    trials = max(1, args.trials)

    # pass 2: opaque the reference ids once, up front
    if args.opaque_ids:
        reference, model_a, model_b = opaquify(case)
    else:
        reference, model_a, model_b = case.reference, case.model_a, case.model_b

    plan = list(conditions(placements, control=not args.no_control))
    total = len(plan) * trials * len(models)

    out = ROOT / "results" / f"ablation_{args.case}{'_opaque' if args.opaque_ids else ''}.csv"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)

    # resume: reload rows already captured for this exact output file (unless --fresh),
    # so a long run can be stopped and continued without redoing completed work
    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {_run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    print(f"Ablation ({'opaque ids' if args.opaque_ids else 'readable ids'}): "
          f"{len(models)} model(s) x {len(plan)} conditions x {trials} trials = {total} runs, "
          f"case {args.case}", file=sys.stderr)

    # open the CSV for incremental capture; header only when starting a fresh file
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
        for placement, flags, use_ref in plan:
            ref_fields = ref_fields_for(flags) if use_ref else None
            label = cell_label(flags) if use_ref else "no-reference"
            stack = None  # created lazily, so fully-skipped conditions cost nothing
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
                                             ref_fields=ref_fields)
                try:
                    rec = stack.reconcile(model_a, model_b,
                                          reference=reference, placement=placement)
                except Exception as e:
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                r = _row(args.case, model, placement, flags, use_ref, t, rec, case.gold, names)
                rows.append(r)
                done.add(key)
                if writer is not None:      # en-route capture: one row, flushed to disk
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
