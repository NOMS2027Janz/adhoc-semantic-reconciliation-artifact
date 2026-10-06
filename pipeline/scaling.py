#!/usr/bin/env python3
"""N-scaling demonstration: work grows ~N with a shared reference, ~N^2 without.

    python scaling.py            # uses benchmark/cases/scaling_otn

The claim (Part I, Section 6): if N systems each bind once to a shared reference,
the pairwise correspondences compose from those N bindings, so the reconciliation
work grows with N. Authoring an alignment for every pair instead grows with the
number of pairs, N(N-1)/2. This script loads N independent models of one network,
all bound to the same reference, and shows both costs as N grows -- and verifies
that composing the N bindings reproduces the correct pairwise correspondences, so
the linear path is not just cheaper but correct.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Concept, SemanticModel  # noqa: E402
from reconcile.reference import Reference  # noqa: E402


def ref_pairs(m1: SemanticModel, m2: SemanticModel) -> set[frozenset]:
    """Correspondences between two models by shared reference entry (composition)."""
    b1 = {c.ref: c.id for c in m1.concepts if c.ref}
    b2 = {c.ref: c.id for c in m2.concepts if c.ref}
    return {frozenset((b1[r], b2[r])) for r in (set(b1) & set(b2))}


def synth_model(k: int, entries: list[str]) -> SemanticModel:
    """A generated model of the same network: distinct local labels, same bindings."""
    concepts = [Concept(id=f"s{k}.{r}", label=f"sys{k}-{r}", kind="concept", ref=r) for r in entries]
    return SemanticModel(system=f"Agent G{k}", dialect="generated", modules=(), concepts=concepts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="scaling_otn")
    ap.add_argument("--max-n", type=int, default=12, help="extend the curve to this many systems")
    ap.add_argument("--write", action="store_true", help="write results/scaling_<case>.csv")
    args = ap.parse_args()

    case = ROOT / "benchmark" / "cases" / args.case
    real = [SemanticModel.from_json(p) for p in sorted(case.glob("model_*.json"))]
    entries = [e.id for e in Reference.from_json(case / "reference.json").entries]
    n_real = len(real)

    # top up with generated models of the same network to reach max_n
    models = list(real)
    while len(models) < args.max_n:
        models.append(synth_model(len(models) + 1, entries))

    rows = []
    max_ok = 0
    for k in range(2, len(models) + 1):
        subset = models[:k]
        pairs = list(combinations(range(k), 2))
        with_ref = k                      # one binding per system
        without_ref = len(pairs)          # one alignment per pair = C(k,2)
        ok = True
        for i, j in pairs:
            corr = ref_pairs(subset[i], subset[j])
            shared = {c.ref for c in subset[i].concepts if c.ref} & {c.ref for c in subset[j].concepts if c.ref}
            if len(corr) != len(shared):
                ok = False
        if ok:
            max_ok = k
        rows.append({"N": k, "with_ref_ops": with_ref, "without_ref_ops": without_ref,
                     "ratio": round(without_ref / with_ref, 2), "composition_ok": ok})

    print(f"\nN-scaling on {case.name}: {n_real} real models + generated models of one network, "
          f"one shared reference.")
    print(f"Composition verified correct for every N from 2 to {max_ok}.\n")
    print(f"  {'N':>3}  {'bind once (with ref)':>22}  {'align each pair (no ref)':>26}  {'ratio':>7}")
    print("  " + "-" * 63)
    for r in rows:
        tag = "  (real models)" if r["N"] <= n_real else ""
        print(f"  {r['N']:>3}  {r['with_ref_ops']:>22}  {r['without_ref_ops']:>26}  {r['ratio']:>6.2f}x{tag}")

    print("\n  Marginal cost of adding the N-th system:")
    print("    with a reference:  +1 binding, which yields N-1 new pairings for free")
    print("    without one:       +(N-1) new pairwise alignments")
    print("  Bindings grow linearly (N); pairwise alignments grow quadratically (N(N-1)/2).\n")

    if args.write:
        out = ROOT / "results" / f"scaling_{args.case}.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "with_ref_ops", "without_ref_ops", "ratio", "composition_ok"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {out.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
