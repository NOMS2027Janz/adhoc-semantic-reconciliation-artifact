#!/usr/bin/env python3
"""E8 -- does correlation survive a DERIVED dependency map? (Brad pt 12.)

The correlation result (E7/E7b) is clean once the resource-dependency map is GIVEN. But in the
study the map is hand-fed. E8 asks the agent to DERIVE it from raw inventory (optical lines lighting
router ports, IP links on ports or LAG bundles, services running over IP links), then tests whether
correlation still works when fed the agent's own derived map instead of the gold one.

Two stages per (model, trial):

  * DERIVE   -- the agent reads the inventory and produces the underlies (dependency) map; scored
               against the gold map (edge precision / recall / exact) from derive_deps_gold.py. The
               inventory contains a LAG-bundle indirection and a name lure, so the derivation is a
               multi-hop join, not a rename.
  * CORRELATE-- run the correlation operation on the same scenarios twice: with the GOLD map (the
               given-map baseline, = E7b) and with the agent's DERIVED map. If derivation is right,
               the two agree; where derivation errs, correlation degrades. This isolates correlation's
               robustness to self-derived structure.

    python pipeline/deps_derivation.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/deps_derivation.py --smoke

Writes results/deps_derivation.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.stacks.agent_observability import CorrelationStack   # noqa: E402
from run import load_dotenv                                         # noqa: E402

CASE = "derive_deps_config"
COLS = ["case", "stage", "model", "trial", "scenario", "arm",
        "map_precision", "map_recall", "map_exact",
        "partition_exact", "cause_accuracy", "reasoning_tokens"]

DERIVE_SYSTEM = (
    "You are deriving a RESOURCE-DEPENDENCY MAP for a layered transport network from its inventory. "
    "'X underlies Y' means X carries or supports Y -- a lower-layer resource beneath a higher-layer "
    "one. The layers here are optical lines, IP links, and services. Rules of the inventory:\n"
    "  * an optical line lights one (router, port);\n"
    "  * an IP link sits on one (router, port); some IP links give a LAG BUNDLE instead of a port, "
    "and you must resolve the bundle to its member port(s) on that router;\n"
    "  * a service runs 'over' one or more named IP links.\n"
    "An optical line underlies an IP link when they occupy the SAME (router, port) -- resolving "
    "bundles first. An IP link underlies a service when the service runs over it. Match on the "
    "physical (router, port), NOT on similar names: two resources with look-alike ids are the same "
    "only if their (router, port) coincide. Return every underlies edge you can justify from the "
    "inventory, and no others."
)


class _Edge(BaseModel):
    resource: str
    underlies: list[str]


class _DepMap(BaseModel):
    edges: list[_Edge]


def _edge_set(mapping: dict) -> set:
    out = set()
    for src, tgts in mapping.items():
        for t in tgts:
            out.add((src, t))
    return out


def derive_map(inventory: dict, model: str, client=None):
    """One model call: inventory -> underlies map (dict) + reasoning tokens."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    payload = {k: inventory[k] for k in ("optical_lines", "ip_links", "bundles", "services")
               if k in inventory}
    t0 = time.time()
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": DERIVE_SYSTEM},
                  {"role": "user", "content": json.dumps(payload, indent=2)}],
        response_format=_DepMap,
    )
    _ = time.time() - t0
    parsed = completion.choices[0].message.parsed
    mapping: dict = {}
    for e in (parsed.edges if parsed else []):
        if e.underlies:
            mapping.setdefault(e.resource, [])
            for u in e.underlies:
                if u not in mapping[e.resource]:
                    mapping[e.resource].append(u)
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    rtok = getattr(details, "reasoning_tokens", None) if details else None
    return mapping, rtok


def score_map(derived: dict, gold: dict) -> dict:
    d, g = _edge_set(derived), _edge_set(gold)
    tp = len(d & g); fp = len(d - g); fn = len(g - d)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"map_precision": round(prec, 3), "map_recall": round(rec, 3),
            "map_exact": (d == g)}


def score_correlation(rec, gincs) -> dict:
    gparts = {frozenset(g["symptoms"]) for g in gincs}
    gcause = {frozenset(g["symptoms"]): g["cause"] for g in gincs}
    aparts = {frozenset(a["symptoms"]) for a in rec.incidents}
    acause = {frozenset(a["symptoms"]): a["cause"] for a in rec.incidents}
    cause_correct = sum(1 for p in gparts if p in acause and acause[p] == gcause[p])
    return {"partition_exact": (aparts == gparts),
            "cause_accuracy": round(cause_correct / len(gincs), 3) if gincs else 0.0}


def run_key(r):
    return (r["stage"], str(r["model"]), str(r["trial"]), r["scenario"], r["arm"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cdir = ROOT / "benchmark" / "cases" / CASE
    inventory = json.loads((cdir / "inventory.json").read_text())
    scenarios = json.loads((cdir / "scenarios.json").read_text())["scenarios"]
    gold_path = cdir / "deps_gold.json"
    if not gold_path.exists():
        print("no deps_gold.json; run benchmark/derive_deps_gold.py first", file=sys.stderr)
        return 2
    gold = json.loads(gold_path.read_text())
    gold_map, gold_inc = gold["underlies"], gold["incidents"]

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / f"{CASE}.csv"
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)
    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        rows = list(csv.DictReader(out.open(newline="")))
        done = {run_key(r) for r in rows}
        print(f"Resuming: {len(done)} rows in {out.name}", file=sys.stderr)
    fh = writer = None
    if write:
        fresh = args.fresh or not out.exists()
        fh = out.open("w" if fresh else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLS)
        if fresh:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    def emit(row):
        full = {c: row.get(c, "") for c in COLS}
        rows.append(full); done.add(run_key(full))
        if writer is not None:
            writer.writerow(full); fh.flush()

    def correlate_arm(model, trial, deps, arm):
        for sc in scenarios:
            key = ("correlate", str(model), str(trial), sc["id"], arm)
            if key in done:
                continue
            scen = dict(sc); scen["deps"] = deps
            stack = CorrelationStack(model=model, pragmatics=True)
            try:
                rec = stack.reconcile(scen)
            except Exception as e:  # noqa: BLE001
                print(f"      ! correlate {arm} {sc['id']}: {e}", file=sys.stderr, flush=True)
                continue
            emit({"case": CASE, "stage": "correlate", "model": model, "trial": trial,
                  "scenario": sc["id"], "arm": arm, **score_correlation(rec, gold_inc[sc["id"]])})

    for model in models:
        for t in range(trials):
            # stage 1: derive the map (skip if already captured)
            derived = None
            dkey = ("derive", str(model), str(t), "", "")
            if dkey not in done:
                print(f"[derive] {model}/trial {t+1}", file=sys.stderr, flush=True)
                try:
                    derived, rtok = derive_map(inventory, model)
                except Exception as e:  # noqa: BLE001
                    print(f"      ! derive: {e}", file=sys.stderr, flush=True)
                    continue
                sm = score_map(derived, gold_map)
                emit({"case": CASE, "stage": "derive", "model": model, "trial": t,
                      "scenario": "", "arm": "", "reasoning_tokens": rtok if rtok is not None else "", **sm})
                print(f"      map precision {sm['map_precision']} recall {sm['map_recall']} exact {sm['map_exact']}",
                      file=sys.stderr, flush=True)
            else:
                # recover the derived map by re-deriving is wasteful; if the derive row exists but we
                # need the map for the derived-correlate arm, re-derive only when that arm is missing
                need_derived_arm = any(("correlate", str(model), str(t), sc["id"], "derived") not in done
                                       for sc in scenarios)
                if need_derived_arm:
                    try:
                        derived, _ = derive_map(inventory, model)
                    except Exception:  # noqa: BLE001
                        derived = None

            # stage 2: correlate with the gold (given) map and with the derived map
            correlate_arm(model, t, gold_map, "given")
            if derived is not None:
                correlate_arm(model, t, derived, "derived")

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    der = [r for r in rows if r["stage"] == "derive"]
    cor = [r for r in rows if r["stage"] == "correlate"]
    if not der and not cor:
        return
    models = []
    for r in der + cor:
        if r["model"] not in models:
            models.append(r["model"])

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, k):
        xs = [num(r[k]) for r in rs if num(r.get(k)) is not None]
        return sum(xs) / len(xs) if xs else None

    def frac_true(rs, k):
        xs = [1.0 if str(r[k]).lower() == "true" else 0.0 for r in rs if str(r.get(k)) != ""]
        return sum(xs) / len(xs) if xs else None

    print("\n=== E8  Dependency-map derivation, and correlation on the derived map ===")
    print(f"    {'model':13}{'map prec':10}{'map rec':9}{'map exact':11}"
          f"{'correlate GIVEN':17}{'correlate DERIVED':18}")
    for m in models:
        d = [r for r in der if r["model"] == m]
        cg = [r for r in cor if r["model"] == m and r["arm"] == "given"]
        cd = [r for r in cor if r["model"] == m and r["arm"] == "derived"]
        mp = mean(d, "map_precision"); mr = mean(d, "map_recall"); mx = frac_true(d, "map_exact")
        pg = frac_true(cg, "partition_exact"); pd = frac_true(cd, "partition_exact")
        def s(v, pct=False):
            if v is None:
                return "-"
            return f"{v:.2f}" if not pct else f"{v:.2f}"
        print(f"    {m:13}{s(mp):10}{s(mr):9}{s(mx):11}{s(pg):17}{s(pd):18}")
    print("\n    Reading: 'map exact' = the agent derived the whole dependency map correctly;")
    print("    'correlate GIVEN' vs 'DERIVED' = exact-incident rate with the hand-fed gold map vs the")
    print("    agent's own map. If DERIVED tracks GIVEN, correlation survives self-derived structure;")
    print("    the gap is the cost of a map the agent had to build rather than be handed.")


if __name__ == "__main__":
    raise SystemExit(main())
