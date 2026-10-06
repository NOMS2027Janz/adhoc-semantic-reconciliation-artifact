#!/usr/bin/env python3
"""E7 -- operation necessity: is each added operation a needed theory extension? (Brad pt 6, D2).

The study exercises three operations that the reference drafts do not name:

  * attribute pinning        -- resolving individuals by pinning on their attributes
  * pragmatic resolution     -- the act/watch/suppress verdict (context-sensitive)
  * composition & correlation-- grouping multi-symptom evidence into incidents with a cause

Brad asks whether each is a genuine THEORY EXTENSION (needed) or a provisional convenience. That
is answerable from runs already on disk: each operation was run with an explicit ON/OFF toggle
across the capability ladder. This driver CONSOLIDATES those runs into one operation x capability
readout -- the correctness delta (ON minus OFF) per checkpoint -- so the question is settled by
evidence, not assertion. It performs NO new model calls; it only reads results/.

Reading rule: an operation whose ablation HURTS (positive ON-OFF delta) is doing real work; a
delta that WIDENS as capability falls means the operation matters most exactly where cognition is
weakest (the reference's job). A delta near zero means the operation is not pulling weight on the
current cases -- a candidate to demote, or a sign the case does not yet stress it.

Sources (already validated, already run):
  * attribute pinning        results/instance_instance_hard_stage1_<model>.csv   (evidence factor 'attrs')
  * pragmatic resolution      results/obs_config_observability_phase3_<model>.csv (pragmatics on/off; verdict_accuracy)
  * composition & correlation results/obs_config_observability_phase4_<model>.csv (pragmatics on/off; partition_exact)

    python operation_necessity.py                 # print the table, write results/operation_necessity.csv
    python operation_necessity.py --no-write
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# capability ladder, strong -> weak
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]
COLUMNS = ["operation", "model", "metric", "on_mean", "off_mean", "delta", "n_on", "n_off",
           "condition"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read(path: Path):
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(newline="")))
    # drop stray re-written header rows (from append/resume) where the key column is literal
    return [r for r in rows if r.get("model") not in (None, "model", "")]


def _mean(rows, key):
    xs = [_num(r[key]) for r in rows if _num(r.get(key)) is not None]
    return (sum(xs) / len(xs)) if xs else None


def _bool_mean(rows, key):
    """Mean of a boolean-ish column ('True'/'False')."""
    xs = [1.0 if str(r.get(key)).strip().lower() == "true" else 0.0
          for r in rows if str(r.get(key)).strip() != ""]
    return (sum(xs) / len(xs)) if xs else None


# ---- attribute pinning: main effect of the 'attrs' evidence factor (instance factorial) -------
def attribute_pinning_rows():
    """Main effect of attribute evidence at both_inert (the reconstruction regime), per model.
    The evidence label lists the ON factors joined by '+'; 'attrs' present vs absent."""
    out = []
    for model in MODELS:
        rows = [r for r in _read(RESULTS / f"instance_instance_hard_stage1_{model}.csv")
                if r.get("placement") == "both_inert"]
        if not rows:
            continue
        def has_attrs(r):
            return "attrs" in (r.get("evidence") or "").split("+")
        on = [r for r in rows if has_attrs(r)]
        off = [r for r in rows if not has_attrs(r)]
        for metric in ("recall", "precision"):
            om, fm = _mean(on, metric), _mean(off, metric)
            if om is None or fm is None:
                continue
            out.append({"operation": "attribute pinning", "model": model, "metric": metric,
                        "on_mean": round(om, 3), "off_mean": round(fm, 3),
                        "delta": round(om - fm, 3), "n_on": len(on), "n_off": len(off),
                        "condition": "both_inert; main effect of 'attrs' over the evidence factorial"})
    return out


# ---- pragmatic resolution: verdict accuracy, pragmatics on vs off (obs phase3) -----------------
def pragmatic_resolution_rows():
    out = []
    for model in MODELS:
        rows = _read(RESULTS / f"obs_config_observability_phase3_{model}.csv")
        if not rows:
            continue
        on = [r for r in rows if r.get("pragmatics") == "on"]
        off = [r for r in rows if r.get("pragmatics") == "off"]
        for metric in ("verdict_accuracy",):
            om, fm = _mean(on, metric), _mean(off, metric)
            if om is None or fm is None:
                continue
            out.append({"operation": "pragmatic resolution", "model": model, "metric": metric,
                        "on_mean": round(om, 3), "off_mean": round(fm, 3),
                        "delta": round(om - fm, 3), "n_on": len(on), "n_off": len(off),
                        "condition": "act/watch/suppress verdict; over contexts normal/maintenance/holiday"})
    return out


# ---- composition & correlation: exact incident partition, pragmatics on vs off (obs phase4) ----
def correlation_rows():
    out = []
    for model in MODELS:
        rows = _read(RESULTS / f"obs_config_observability_phase4_{model}.csv")
        if not rows:
            continue
        on = [r for r in rows if r.get("pragmatics") == "on"]
        off = [r for r in rows if r.get("pragmatics") == "off"]
        specs = [("partition_exact", _bool_mean), ("cause_accuracy", _mean)]
        for metric, agg in specs:
            om, fm = agg(on, metric), agg(off, metric)
            if om is None or fm is None:
                continue
            out.append({"operation": "composition & correlation", "model": model, "metric": metric,
                        "on_mean": round(om, 3), "off_mean": round(fm, 3),
                        "delta": round(om - fm, 3), "n_on": len(on), "n_off": len(off),
                        "condition": "incident partition over scenarios S1/S2/S3"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    rows = attribute_pinning_rows() + pragmatic_resolution_rows() + correlation_rows()
    if not rows:
        print("No source result files found under results/. Nothing to consolidate.", file=sys.stderr)
        return 1

    # report: one block per operation, ladder strong->weak, primary metric first
    print("\n=== E7  Operation necessity: correctness ON minus OFF, across the capability ladder ===")
    print("    positive delta = the operation is doing real work; a delta that widens as the")
    print("    model weakens = the operation matters most where cognition is weakest.\n")
    by_op = {}
    for r in rows:
        by_op.setdefault(r["operation"], []).append(r)
    order = ["attribute pinning", "pragmatic resolution", "composition & correlation"]
    for op in order:
        ors = by_op.get(op, [])
        if not ors:
            continue
        metrics = []
        for r in ors:
            if r["metric"] not in metrics:
                metrics.append(r["metric"])
        primary = metrics[0]
        print(f"  {op}   (primary metric: {primary})")
        print(f"    {'model':13}{'metric':20}{'ON':>7}{'OFF':>8}{'delta':>8}   n(on/off)")
        for model in MODELS:
            for metric in metrics:
                sel = [r for r in ors if r["model"] == model and r["metric"] == metric]
                if not sel:
                    continue
                r = sel[0]
                star = "  <-- primary" if metric == primary else ""
                print(f"    {model:13}{metric:20}{r['on_mean']:>7}{r['off_mean']:>8}"
                      f"{r['delta']:>+8}   {r['n_on']}/{r['n_off']}{star}")
        # ladder trend on the primary metric
        prim = [r for r in ors if r["metric"] == primary]
        prim_by_model = {r["model"]: r for r in prim}
        deltas = [(m, prim_by_model[m]["delta"]) for m in MODELS if m in prim_by_model]
        if len(deltas) >= 2:
            trend = "widens as capability falls" if deltas[-1][1] > deltas[0][1] + 1e-9 else \
                    ("narrows as capability falls" if deltas[-1][1] < deltas[0][1] - 1e-9 else "flat across the ladder")
            print(f"    -> primary-metric delta {deltas[0][1]:+} (strong) .. {deltas[-1][1]:+} (weak): {trend}")
        print()

    if not args.no_write:
        out = RESULTS / "operation_necessity.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
