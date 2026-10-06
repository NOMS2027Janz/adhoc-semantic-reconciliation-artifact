#!/usr/bin/env python3
"""E4' -- escalation and probe usage: when do agents fall back to the decisive experiment?
(Brad pt 3, reshaped.)

The original E4 (a misleading-key over-trust test) is impossible in this benchmark: the instance
gold is derived so that keys never collide across truths -- a key cannot lie -- so there is no
key-vs-attributes conflict to stage. Reshaped, the live question Brad pt 3 raises is the fallback
ORDERING: when cheaper evidence (the key) is withheld, does the agent escalate to the more
expensive decisive experiment (interrogate / provision probes), and is that escalation effective?
That is answerable from the instance runs already on disk -- they record probe counts, turns, and
recall per evidence condition, placement, budget and model. This driver consolidates them. No new
model calls.

Two readouts:

  A. Escalation TRIGGER (instance stage 1, live placements). Probe usage and recall with the key
     PRESENT vs ABSENT, per capability. Does withholding the key make the agent probe more, and
     does the extra probing recover the lost recall?

  B. Escalation PAYOFF and its REACH (instance stage 3, oracle-budget sweep {0, 3, unbounded}).
     How much recall does a probe budget buy, and does it pay off only where a side is live
     (both_cognitive) versus where the experiment has no reach (one_inert)?

    python pipeline/escalation_analysis.py            # print tables, write results/escalation_analysis.csv
    python pipeline/escalation_analysis.py --no-write
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ["gpt-5.6-sol", "gpt-5-mini", "gpt-5-nano"]   # strong -> weak
COLUMNS = ["readout", "model", "condition", "placement", "recall", "experiment_only_recall",
           "probes_per_run", "turns", "n"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read(path: Path):
    if not path.exists():
        return []
    return [r for r in csv.DictReader(path.open(newline="")) if r.get("placement") not in (None, "placement", "")]


def _mean(rows, key):
    xs = [_num(r[key]) for r in rows if _num(r.get(key)) is not None]
    return (sum(xs) / len(xs)) if xs else None


def _probes(rows):
    ic = _mean(rows, "interrogate_calls") or 0.0
    pc = _mean(rows, "provision_calls") or 0.0
    return ic + pc


def _has_key(ev: str) -> bool:
    return "key" in (ev or "").split("+")


# ---- A. escalation trigger: probe usage vs key availability, live placements (stage 1) --------
def trigger_rows():
    out = []
    for model in MODELS:
        rows = _read(RESULTS / f"instance_instance_hard_stage1_{model}.csv")
        for pl in ("both_cognitive", "one_inert"):
            prs = [r for r in rows if r.get("placement") == pl]
            for label, keep in (("key present", lambda r: _has_key(r.get("evidence", ""))),
                                ("key absent", lambda r: not _has_key(r.get("evidence", "")))):
                rs = [r for r in prs if keep(r)]
                if not rs:
                    continue
                out.append({"readout": "trigger", "model": model, "condition": label, "placement": pl,
                            "recall": round(_mean(rs, "recall") or 0, 3),
                            "experiment_only_recall": round(_mean(rs, "experiment_only_recall") or 0, 3),
                            "probes_per_run": round(_probes(rs), 2),
                            "turns": round(_mean(rs, "turns") or 0, 2), "n": len(rs)})
    return out


# ---- B. escalation payoff and reach: budget sweep (stage 3) ------------------------------------
BUDGET_ORDER = {"0": 0, "3": 1, "unbounded": 2}


def payoff_rows():
    out = []
    for model in MODELS:
        rows = _read(RESULTS / f"instance_instance_hard_stage3_{model}.csv")
        if not rows:
            continue
        for pl in ("both_cognitive", "one_inert"):
            for bud in sorted({r.get("budget") for r in rows if r.get("placement") == pl},
                              key=lambda b: BUDGET_ORDER.get(b, 9)):
                rs = [r for r in rows if r.get("placement") == pl and r.get("budget") == bud]
                if not rs:
                    continue
                out.append({"readout": "payoff", "model": model, "condition": f"budget={bud}", "placement": pl,
                            "recall": round(_mean(rs, "recall") or 0, 3),
                            "experiment_only_recall": round(_mean(rs, "experiment_only_recall") or 0, 3),
                            "probes_per_run": round(_probes(rs), 2),
                            "turns": round(_mean(rs, "turns") or 0, 2), "n": len(rs)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    trig = trigger_rows()
    pay = payoff_rows()
    if not trig and not pay:
        print("No instance stage1/stage3 result files found under results/.", file=sys.stderr)
        return 1

    print("\n=== E4'  A. Escalation TRIGGER: does withholding the key make the agent probe more? ===")
    print("    instance stage 1, live placements; probes = interrogate + provision calls per run\n")
    print(f"    {'model':13}{'placement':16}{'key':14}{'recall':8}{'probes/run':12}{'turns':7}")
    for model in MODELS:
        for pl in ("both_cognitive", "one_inert"):
            sel = [r for r in trig if r["model"] == model and r["placement"] == pl]
            for cond in ("key present", "key absent"):
                r = next((x for x in sel if x["condition"] == cond), None)
                if r:
                    print(f"    {model:13}{pl:16}{cond:14}{r['recall']:<8}{r['probes_per_run']:<12}{r['turns']:<7}")
        # the escalation ratio on both_cognitive: probing response to the missing key
        bc = {r["condition"]: r for r in trig if r["model"] == model and r["placement"] == "both_cognitive"}
        if "key present" in bc and "key absent" in bc and bc["key present"]["probes_per_run"] > 0:
            ratio = bc["key absent"]["probes_per_run"] / bc["key present"]["probes_per_run"]
            dr = bc["key absent"]["recall"] - bc["key present"]["recall"]
            print(f"      -> {model}: key absent probes x{ratio:.1f} vs present; recall change {dr:+.3f}")
        print()

    print("=== E4'  B. Escalation PAYOFF and REACH: what a probe budget buys, by placement ===")
    print("    instance stage 3, oracle-budget sweep; payoff is reach-gated by a live side\n")
    print(f"    {'model':13}{'placement':16}{'budget':18}{'recall':8}{'exp-only':10}{'probes/run':12}")
    for model in MODELS:
        rs = [r for r in pay if r["model"] == model]
        if not rs_present(rs):
            continue
        for pl in ("both_cognitive", "one_inert"):
            for r in sorted([x for x in rs if x["placement"] == pl],
                            key=lambda x: BUDGET_ORDER.get(x["condition"].split("=")[1], 9)):
                print(f"    {model:13}{pl:16}{r['condition']:18}{r['recall']:<8}"
                      f"{r['experiment_only_recall']:<10}{r['probes_per_run']:<12}")
        print()
    print("    Reading: escalation to the decisive experiment recovers recall the withheld key")
    print("    would have supplied -- but the escalation is capability-gated (a weak model does not")
    print("    ramp its probing when the key vanishes) and reach-gated (probes buy little at one_inert,")
    print("    where no fully live side can be interrogated -- the instance echo of E6).")

    if not args.no_write:
        out = RESULTS / "escalation_analysis.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in trig + pay:
                w.writerow(r)
        print(f"\nWrote {out.relative_to(ROOT)} ({len(trig) + len(pay)} rows)")
    return 0


def rs_present(rs):
    return bool(rs)


if __name__ == "__main__":
    raise SystemExit(main())
