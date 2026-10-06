#!/usr/bin/env python3
"""E18 -- the minimum viable overlay: how little authored overlay rescues a thin source?

E16 showed the governed overlay is the robust route: with the overlay on, even an inert (thin)
source reconciles across the capability ladder. The obvious next objection is cost -- "definitions
are valuable, but we are not going to go back and change every existing interface model" (Brad). So
this quantifies the authoring cost: over a THIN source, how little overlay is enough, on two axes?

  COVERAGE   how many concepts need an authored entry at all. The overlay covers a shrinking subset
             of the five seam categories, dropped EASY-first (the label-matched rate/latency/
             protection) and keeping the HARD, non-obvious seams last (hand-off<->attachment, then
             circuit<->underlay). Answers "author a thin supplement, not a re-model" with a number:
             which entries are load-bearing.
  RICHNESS   how much each entry must carry. At full coverage, each entry is thinned from the full
             field set down to id-only (reusing the reference-ablation field masks, E5), over the
             thin source rather than E5's described one.

Construction reuses the ad-hoc reference of `config_cross_domain` with OPAQUE ids (so an id carries
no meaning and the descriptive fields are the only binding evidence). The source side (Meridian) is
presented INERT (label, kind, relations, instances -- no gloss); the target side (Cascade) is live
and carries its overlay binding only where the overlay covers it. Reconciled at placement one_inert
via the native agent stack. Coverage level 0 is the no-overlay floor (the E16 inert-off baseline).

Reported per level/model: precision, recall, seam recall (the two hard seams), whether the grade
cognate was taken. Across the capability ladder.

Prediction. Recovery of the non-obvious seams tracks whether THEIR entry is present: you can drop the
easy entries cheaply, but the hard seams need their overlay entry over a thin source (the labels do
not align). On richness, a single descriptive field per entry (the definition) should do most of the
work; id-only cannot bind. The "number": far fewer than one-entry-per-concept, concentrated on the
ambiguous seams.

Usage (run from study/; Chris runs it):
    python pipeline/minimum_overlay.py --trials 3
    python pipeline/minimum_overlay.py --trials 3 --models gpt-5.6-sol --out results/e18_sol.csv
    python pipeline/minimum_overlay.py --validate        # offline construction + scoring, no API

Writes results/minimum_overlay.csv (one row per model/sweep-level/trial).
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
from reconcile.model import Case                 # noqa: E402
from reconcile.reference import Reference        # noqa: E402
from reconcile.metrics import score              # noqa: E402
from run import load_dotenv                       # noqa: E402

CASE = "config_cross_domain"
HARD_SEAMS = [frozenset(("m.circuit", "c.underlay")), frozenset(("m.handoff", "c.attachment"))]

# the five seam categories (reference entry ids), in DROP order: easy (label-matched) first, the
# non-obvious seams kept last. Coverage level k keeps the HARDEST k of these.
DROP_ORDER = ["service-rate", "latency-bound", "protection-requirement", "demarcation",
              "transport-underlay"]

# richness ladder at full coverage: field-mask per entry (subset of REF_FIELDS_ALL). id is always
# present (the coreference anchor); the empty set is the id-only floor.
FIELD_LADDER = [("id_only", set()), ("label", {"label"}), ("definition", {"definition"}),
                ("def_example", {"definition", "example"}),
                ("full", {"label", "synonyms", "class", "definition", "example"})]


def opaquify(case):
    """Replace every reference id with an opaque token (e01, ...) and remap each concept's binding,
    so the id carries no meaning and the descriptive fields are the only binding evidence. Returns
    (opaque_reference, model_a, model_b, idmap: original_id -> opaque_id)."""
    ref = case.reference
    idmap = {e.id: f"e{i:02d}" for i, e in enumerate(ref.entries, 1)}
    new_ref = dataclasses.replace(
        ref, entries=[dataclasses.replace(e, id=idmap[e.id]) for e in ref.entries])

    def remap(m):
        return dataclasses.replace(
            m, concepts=[dataclasses.replace(c, ref=idmap.get(c.ref, c.ref)) for c in m.concepts])

    return new_ref, remap(case.model_a), remap(case.model_b), idmap


def coverage_keep(k):
    """The seam categories present at coverage level k: the HARDEST k (drop easy-first)."""
    return set(DROP_ORDER[len(DROP_ORDER) - k:]) if k > 0 else set()


def build_coverage(base_ref, mb, idmap, k):
    """(reference, model_b) for coverage level k: overlay carries entries only for the kept
    categories, and the live side binds only where an entry exists."""
    keep_opaque = {idmap[c] for c in coverage_keep(k)}
    entries = [e for e in base_ref.entries if e.id in keep_opaque]
    ref = dataclasses.replace(base_ref, entries=entries)
    mb_k = dataclasses.replace(
        mb, concepts=[dataclasses.replace(c, ref=(c.ref if c.ref in keep_opaque else None))
                      for c in mb.concepts])
    return ref, mb_k


COLUMNS = ["case", "model", "sweep", "level", "n_entries", "fields", "uses_reference", "trial",
           "precision", "recall", "seam_recall", "took_grade_cognate",
           "surviving_false_cognates", "residual", "reasoning_tokens", "missed"]


def _row(model, sweep, level, n_entries, fields, use_ref, trial, rec, gold):
    m = score(rec, gold)
    proposed = set(rec.proposed)
    missed = gold.correct_pairs - proposed
    seam_hit = sum(1 for s in HARD_SEAMS if s in proposed)
    return {
        "case": CASE, "model": str(model), "sweep": sweep, "level": str(level),
        "n_entries": n_entries, "fields": fields, "uses_reference": str(bool(use_ref)),
        "trial": trial, "precision": m["precision"], "recall": m["recall"],
        "seam_recall": round(seam_hit / len(HARD_SEAMS), 3),
        "took_grade_cognate": int(len(proposed & gold.false_cognate_pairs) > 0),
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "missed": ";".join(sorted("~".join(sorted(p)) for p in missed)),
    }


def _run_key(r):
    return (str(r["model"]), r["sweep"], str(r["level"]), str(r["trial"]))


def conditions():
    """Yield (sweep, level, n_entries, fields, use_ref, ref_fields_mask). Coverage 0..5 at full
    richness; then the richness ladder at full coverage (skipping 'full', which the coverage sweep's
    level-5 already is)."""
    full_mask = {"label", "synonyms", "class", "definition", "example"}
    for k in range(0, 6):
        yield ("coverage", k, k, "full" if k > 0 else "none", k > 0, full_mask)
    for name, mask in FIELD_LADDER:
        if name == "full":
            continue
        yield ("fields", name, 5, name, True, mask)


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    for model in models:
        mr = [r for r in rows if r["model"] == model]
        if not mr:
            continue
        print(f"\n{'#'*10} {model} {'#'*10}")
        for sweep, head in (("coverage", "overlay coverage (entries kept, hardest-last)"),
                            ("fields", "entry richness at full coverage")):
            sel = [r for r in mr if r["sweep"] == sweep]
            if not sel:
                continue
            print(f"  {head}")
            print("    level            entries  precision  recall  seam  grade-cognate")
            levels = [str(x) for x in range(6)] if sweep == "coverage" \
                else [n for n, _ in FIELD_LADDER if n != "full"]
            for lv in levels:
                s = [r for r in sel if str(r["level"]) == str(lv)]
                if not s:
                    continue
                def mn(k):
                    xs = [float(x[k]) for x in s if x[k] not in ("", None)]
                    return sum(xs) / len(xs) if xs else 0.0
                print(f"    {str(lv):<15}  {s[0]['n_entries']:>6}   {mn('precision'):.2f}"
                      f"       {mn('recall'):.2f}    {mn('seam_recall'):.2f}   {mn('took_grade_cognate'):.2f}")
    print()


def validate() -> int:
    case = Case.load(ROOT / "benchmark" / "cases" / CASE)
    base_ref, ma, mb, idmap = opaquify(case)
    gold = case.gold
    ok = True
    print(f"E18 construction check ({CASE}):")
    print(f"  reference entries opaqued: {[e.id for e in base_ref.entries]}")
    seams_ok = all(s in gold.correct_pairs for s in HARD_SEAMS)
    print(f"  hard seams in gold: {seams_ok}")
    ok = ok and seams_ok
    # coverage: entry counts and that the kept set is the hardest-k, live side bound only where kept
    for k in range(0, 6):
        ref, mb_k = build_coverage(base_ref, mb, idmap, k)
        kept_cats = coverage_keep(k)
        bound = sum(1 for c in mb_k.concepts if c.ref is not None)
        entries_ok = len(ref.entries) == k
        # at k>=1 the hardest seam (transport-underlay) must be kept; at k>=2 also demarcation
        hard_ok = (("transport-underlay" in kept_cats) if k >= 1 else True) and \
                  (("demarcation" in kept_cats) if k >= 2 else True)
        print(f"  cov k={k}: entries={len(ref.entries)} kept={sorted(kept_cats)} "
              f"live-bound-concepts={bound}  entries_ok={entries_ok} hard_last_ok={hard_ok}")
        ok = ok and entries_ok and hard_ok and (bound == k)
    # richness masks at full coverage
    for name, mask in FIELD_LADDER:
        print(f"  fields[{name:<11}] mask={sorted(mask)}")
    # scoring self-check (needs pydantic via agent_openai; skip gracefully where it is absent, e.g.
    # the device's system python, so the construction checks above still run there)
    try:
        from reconcile.stacks.agent_openai import OpenAIAgentStack, ReconcileResult, _Correspondence
    except ModuleNotFoundError as e:
        print(f"  (scoring self-check skipped: {e}; run --validate under the venv to exercise it)")
        print("OK" if ok else "FAILED")
        return 0 if ok else 1
    stack = OpenAIAgentStack(use_reference=True, model="mock", inert_side="a")
    corr = [_Correspondence(a_id=c["a"], b_id=c["b"], confidence=0.9, rationale="x")
            for c in gold.correspondences]
    result = ReconcileResult(correspondences=corr, residual_a=[], residual_b=[])
    rec = stack.to_reconciliation(result, ma, mb, {"reasoning_tokens": 0}, "one_inert")
    r = _row("mock", "coverage", 5, 5, "full", True, 0, rec, gold)
    print(f"  scoring self-check (all seams proposed): precision={r['precision']} "
          f"recall={r['recall']} seam_recall={r['seam_recall']} grade={r['took_grade_cognate']}")
    ok = ok and (r["recall"] == 1.0 and r["seam_recall"] == 1.0 and r["took_grade_cognate"] == 0)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="results/minimum_overlay.csv")
    ap.add_argument("--validate", action="store_true",
                    help="offline: check the coverage/richness construction and scoring, no API")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.validate:
        return validate()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack

    case = Case.load(ROOT / "benchmark" / "cases" / CASE)
    base_ref, ma, mb, idmap = opaquify(case)
    gold = case.gold
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    trials = max(1, args.trials)
    plan = list(conditions())

    out = ROOT / args.out
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {_run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    total = len(models) * len(plan) * trials
    print(f"E18 minimum overlay: {len(models)} model(s) x {len(plan)} levels x {trials} trials "
          f"= {total} runs", file=sys.stderr)

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
        for sweep, level, n_entries, fields, use_ref, mask in plan:
            # build the (reference, model_b) for this level
            if sweep == "coverage":
                ref, mb_use = build_coverage(base_ref, mb, idmap, level)
            else:
                ref, mb_use = base_ref, mb
            ref_fields = mask if use_ref else None
            for t in range(trials):
                i += 1
                key = (str(model), sweep, str(level), str(t))
                if key in done:
                    print(f"[{i}/{total}] {model} / {sweep}:{level} / trial {t+1}  (captured, skip)",
                          file=sys.stderr, flush=True)
                    continue
                print(f"[{i}/{total}] {model} / {sweep}:{level} / trial {t+1}",
                      file=sys.stderr, flush=True)
                try:
                    stack = OpenAIAgentStack(use_reference=use_ref, model=model,
                                             inert_side="a", ref_fields=ref_fields)
                    rec = stack.reconcile(ma, mb_use, reference=(ref if use_ref else None),
                                          placement="one_inert")
                except Exception as e:
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                r = _row(model, sweep, level, n_entries, fields, use_ref, t, rec, gold)
                rows.append(r)
                done.add(key)
                if writer is not None:
                    writer.writerow(r)
                    fh.flush()
                print(f"      prec {r['precision']} rec {r['recall']} seam {r['seam_recall']} "
                      f"grade {r['took_grade_cognate']} missed[{r['missed']}]",
                      file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
