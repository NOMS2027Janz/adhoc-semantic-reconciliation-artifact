#!/usr/bin/env python3
"""E14 -- the lexicon-divergence sweep: where does the false-cognate guard break?

E5 established a *negative*: thinning the reference's descriptive fields does not move
the strong model's precision, so field-richness is not the reference-quality axis. What
E5 pointed to instead -- and what E1 named as its next step -- is LEXICON DIVERGENCE: how
differently the two sides name and carve up the same reality. This experiment locates the
threshold on that axis.

The construction reuses the two endpoints we already own, which carry the SAME gold:

    config_evpn            -- both sides bridge to ONE shared reference entry per concept
                              (a shared-id bridge; divergence d = 0)
    config_evpn_indeplex   -- each side carries its OWN independently authored lexicon,
                              no id shared across sides (E1; divergence d = 1)

The concept surfaces (labels, glosses, relations, instances) are IDENTICAL across the two;
the only thing that differs is the reference regime -- whether a corresponding pair shares
one bridge entry, or must be aligned across two independent lexicon entries by meaning. So
we can sweep divergence with everything else fixed: for a level with p independently
lexicalised pairs (p = 0, 2, 4, 6, 8 of the 8 correspondences), the first p pairs in a
fixed order carry independent lexicon entries and the rest keep the shared bridge; gold is
invariant.

The two planted cross-lexicon false cognates cross into the independent (hard) regime at
known levels: the "segment" trap (m.seg vs e.esi) at p = 2, and the "MAC-" trap
(m.maclearn vs e.macmob) at p = 6. So the sweep does not just trace a divergence fraction;
it resolves WHICH concepts becoming independent drives the precision loss -- locating the
threshold at the false-cognate-bearing concepts rather than at an overall divergence level.

To isolate divergence as the ONLY variable, the agent instruction is held fixed at the
independent-lexicon note for every level (the agent is never told to trust ids), so p = 0
is "all bridges shared, but alignment still asked for" -- which should still reach precision
1.0. The note-matched endpoints (config_evpn under the shared note, config_evpn_indeplex
under the independent note) are E1's already-reported anchors.

    python divergence_sweep.py --trials 4                         # core: both_inert, ref on
    python divergence_sweep.py --trials 4 --models gpt-5.6-sol    # one checkpoint
    python divergence_sweep.py --validate                          # offline: check construction only

Writes results/divergence_sweep.csv (one row per run).
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

# ---------------------------------------------------------------------------
# the divergence ladder: a fixed order in which correspondence pairs cross from
# the shared-bridge regime to the independent-lexicon regime. The two trap-bearing
# pairs are placed so the "segment" trap crosses at p=2 and the "MAC-" trap at p=6.
# Each pair also carries the residual decoy / self-extend concept that shares its
# reference regime (e.esi with seg, e.macmob with maclearn, m.sla with ethsvc).
# ---------------------------------------------------------------------------
DIVERGE_ORDER = [
    {"name": "seg",      "a": ["m.seg"],       "b": ["e.vlan", "e.esi"]},     # trap 1 crosses here (p>=2)
    {"name": "uni",      "a": ["m.uni"],       "b": ["e.ac"]},
    {"name": "ethsvc",   "a": ["m.ethsvc", "m.sla"], "b": ["e.evi"]},
    {"name": "ep",       "a": ["m.ep"],        "b": ["e.access"]},
    {"name": "maclearn", "a": ["m.maclearn"],  "b": ["e.maclearn", "e.macmob"]},  # trap 2 crosses here (p>=6)
    {"name": "bwp",      "a": ["m.bwp"],       "b": ["e.policer"]},
    {"name": "cos",      "a": ["m.cos"],       "b": ["e.tc"]},
    {"name": "mtu",      "a": ["m.mtu"],       "b": ["e.mtu"]},
]

# concept id -> its independent-lexicon reference id (from config_evpn_indeplex)
INDEP_A = {"m.seg": "la.bcast", "m.uni": "la.uni", "m.ethsvc": "la.evc", "m.sla": "la.avail",
           "m.ep": "la.evcep", "m.maclearn": "la.macl", "m.bwp": "la.bwp", "m.cos": "la.cos",
           "m.mtu": "la.mtu"}
INDEP_B = {"e.vlan": "lb.bd", "e.esi": "lb.eseg", "e.ac": "lb.acif", "e.evi": "lb.evi",
           "e.access": "lb.vna", "e.maclearn": "lb.macl", "e.macmob": "lb.macmob",
           "e.policer": "lb.policer", "e.tc": "lb.tclass", "e.mtu": "lb.mtu"}

LEVELS = [0, 2, 4, 6, 8]          # number of independently lexicalised pairs (of 8)
N_PAIRS = 8

COLUMNS = ["case", "model", "placement", "level_pairs", "divergence", "uses_reference",
           "trial", "precision", "recall", "surviving_false_cognates", "residual",
           "confident_errors", "confident_error_rate", "conf_wrong_mean", "calibration_gap",
           "reasoning_tokens", "surviving_traps"]


def _load_entry_index():
    """Merge the shared reference entries (config_evpn) and the independent lexicon entries
    (config_evpn_indeplex) into one id -> ReferenceEntry index. The two id spaces are
    disjoint (shared ids vs la.*/lb.*), so the merge is unambiguous."""
    shared = Reference.from_json(ROOT / "benchmark" / "cases" / "config_evpn" / "reference.json")
    indep = Reference.from_json(ROOT / "benchmark" / "cases" / "config_evpn_indeplex" / "reference.json")
    idx = {e.id: e for e in shared.entries}
    idx.update({e.id: e for e in indep.entries})
    return idx


def _flips(p: int) -> tuple[set, set]:
    """The A-side and B-side concept ids that are independently lexicalised at level p."""
    flip_a: set = set()
    flip_b: set = set()
    for pair in DIVERGE_ORDER[:p]:
        flip_a.update(pair["a"])
        flip_b.update(pair["b"])
    return flip_a, flip_b


def build_level(base: Case, entry_index: dict, p: int):
    """Return (model_a, model_b, reference) for divergence level p, built from the shared
    base case by re-pointing the first p pairs' concepts at their independent lexicon
    entries. Concept surfaces are untouched; only `ref` changes, and the reference payload
    is assembled from whichever entries the concepts now point at."""
    flip_a, flip_b = _flips(p)

    def repoint(model, flips, indep_map):
        new_concepts = []
        for c in model.concepts:
            if c.id in flips and c.id in indep_map:
                new_concepts.append(dataclasses.replace(c, ref=indep_map[c.id]))
            else:
                new_concepts.append(c)
        return dataclasses.replace(model, concepts=new_concepts)

    a = repoint(base.model_a, flip_a, INDEP_A)
    b = repoint(base.model_b, flip_b, INDEP_B)

    used = []
    seen = set()
    for c in list(a.concepts) + list(b.concepts):
        if c.ref and c.ref not in seen:
            seen.add(c.ref)
            if c.ref not in entry_index:
                raise SystemExit(f"level p={p}: no reference entry for id {c.ref!r}")
            used.append(entry_index[c.ref])
    # kind='independent' for every level: the agent is told to align by meaning throughout,
    # so divergence (how many pairs actually need aligning) is the only variable.
    reference = Reference(id=f"ref.divergence.p{p}", kind="independent", entries=used)
    return a, b, reference


def trap_names(gold):
    out = {}
    for fc in gold.false_cognates:
        a, b = fc["a"], fc["b"]
        out[frozenset((a, b))] = f"{a}~{b}"
    return out


def surviving_traps(rec, gold, names) -> str:
    hit = set(rec.proposed) & gold.false_cognate_pairs
    return ";".join(sorted(names.get(p, "?") for p in hit))


def _row(case, model, placement, p, use_ref, trial, rec, gold, names):
    m = score(rec, gold)
    return {
        "case": case, "model": str(model), "placement": placement,
        "level_pairs": p, "divergence": round(p / N_PAIRS, 3),
        "uses_reference": str(use_ref), "trial": trial,
        "precision": m["precision"], "recall": m["recall"],
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "confident_errors": m["confident_errors"], "confident_error_rate": m["confident_error_rate"],
        "conf_wrong_mean": m["conf_wrong_mean"], "calibration_gap": m["calibration_gap"],
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "surviving_traps": surviving_traps(rec, gold, names),
    }


def _run_key(r):
    return (str(r["model"]), r["placement"], str(r["level_pairs"]),
            str(r["uses_reference"]), str(r["trial"]))


def conditions(placements, ref_modes):
    """Yield (placement, p, use_reference). The no-reference floor is (near) constant across
    levels -- the models are identical without the reference -- so it is run only at the two
    endpoints as a validity check, not at every level."""
    for placement in placements:
        for use_ref in ref_modes:
            levels = LEVELS if use_ref else [0, 8]
            for p in levels:
                yield placement, p, use_ref


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    for model in models:
        mr = [r for r in rows if r["model"] == model and r["uses_reference"] == "True"
              and r["placement"] == "both_inert"]
        if not mr:
            continue
        print(f"\n{'#'*10} {model} (both_inert, reference on) {'#'*10}")
        print("  divergence   precision   surv.false-cog   confident-errors   traps-taken")
        for p in LEVELS:
            sel = [r for r in mr if r["level_pairs"] == p]
            if not sel:
                continue
            def mean(k):
                xs = [float(x[k]) for x in sel if x[k] not in ("", None)]
                return sum(xs) / len(xs) if xs else 0.0
            traps = sorted({t for r in sel for t in r["surviving_traps"].split(";") if t})
            print(f"   d={p/N_PAIRS:.2f} (p={p})   "
                  f"{mean('precision'):.2f}        {mean('surviving_false_cognates'):.2f}"
                  f"              {mean('confident_errors'):.2f}            [{','.join(traps)}]")
    print()


# ---------------------------------------------------------------------------
# offline validation: build every level and check the construction is sound and the
# endpoints reproduce the two existing cases -- no API, no cost.
# ---------------------------------------------------------------------------
def validate() -> int:
    base = Case.load(ROOT / "benchmark" / "cases" / "config_evpn")
    indeplex = Case.load(ROOT / "benchmark" / "cases" / "config_evpn_indeplex")
    idx = _load_entry_index()
    ok = True
    print("Divergence-ladder construction check:")
    for p in LEVELS:
        a, b, ref = build_level(base, idx, p)
        arefs = {c.id: c.ref for c in a.concepts}
        brefs = {c.id: c.ref for c in b.concepts}
        n_indep = sum(1 for v in list(arefs.values()) + list(brefs.values())
                      if v and (v.startswith("la.") or v.startswith("lb.")))
        print(f"  p={p} (d={p/N_PAIRS:.2f}): {len(ref.entries)} reference entries, "
              f"{n_indep} concepts on an independent lexicon")
        # endpoint checks
        if p == 0:
            same = all(arefs[c.id] == c.ref for c in base.model_a.concepts) and \
                   all(brefs[c.id] == c.ref for c in base.model_b.concepts)
            print(f"    endpoint p=0 reproduces config_evpn refs: {same}")
            ok = ok and same
        if p == 8:
            same = all(arefs[c.id] == c.ref for c in indeplex.model_a.concepts) and \
                   all(brefs[c.id] == c.ref for c in indeplex.model_b.concepts)
            eset = {e.id for e in ref.entries}
            iset = {e.id for e in indeplex.reference.entries}
            print(f"    endpoint p=8 reproduces config_evpn_indeplex refs: {same}; "
                  f"reference entry set matches: {eset == iset}")
            ok = ok and same and (eset == iset)
    # gold invariance / trap presence
    g = base.gold
    print(f"  gold: {len(g.correspondences)} correspondences, "
          f"{len(g.false_cognates)} false cognates {sorted(trap_names(g).values())}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano",
                    help="comma-separated model list (the capability ladder)")
    ap.add_argument("--placements", default="both_inert",
                    help="comma-separated; both_inert is where the reference does the binding")
    ap.add_argument("--ref-modes", default="on,off",
                    help="'on,off' runs the reference-on sweep plus a no-reference floor at the endpoints")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="results/divergence_sweep.csv",
                    help="output CSV (relative to study/); give each parallel terminal its own "
                         "file, e.g. results/divergence_sweep_sol.csv, then merge")
    ap.add_argument("--validate", action="store_true",
                    help="offline: check the ladder construction and endpoints, no API calls")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results file and start over (default: resume)")
    args = ap.parse_args()

    if args.validate:
        return validate()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack

    base = Case.load(ROOT / "benchmark" / "cases" / "config_evpn")
    idx = _load_entry_index()
    names = trap_names(base.gold)
    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ref_modes = [m.strip() == "on" for m in args.ref_modes.split(",") if m.strip()]
    trials = max(1, args.trials)

    plan = list(conditions(placements, ref_modes))
    total = len(plan) * trials * len(models)

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

    print(f"Divergence sweep: {len(models)} model(s) x {len(plan)} conditions x {trials} trials "
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
        for placement, p, use_ref in plan:
            a, b, reference = build_level(base, idx, p)
            stack = None
            for t in range(trials):
                i += 1
                key = (str(model), placement, str(p), str(use_ref), str(t))
                tag = f"d={p/N_PAIRS:.2f} ref={'on' if use_ref else 'off'} {placement}"
                if key in done:
                    print(f"[{i}/{total}] {model} / {tag} / trial {t+1}  (captured, skip)",
                          file=sys.stderr, flush=True)
                    continue
                print(f"[{i}/{total}] {model} / {tag} / trial {t+1}", file=sys.stderr, flush=True)
                if stack is None:
                    stack = OpenAIAgentStack(use_reference=use_ref, model=model)
                try:
                    rec = stack.reconcile(a, b, reference=reference, placement=placement)
                except Exception as e:
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                r = _row(base.name, model, placement, p, use_ref, t, rec, base.gold, names)
                rows.append(r)
                done.add(key)
                if writer is not None:
                    writer.writerow(r)
                    fh.flush()
                print(f"      prec {r['precision']} rec {r['recall']} "
                      f"surv.fc {r['surviving_false_cognates']} conf.err {r['confident_errors']} "
                      f"traps[{r['surviving_traps']}]", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
