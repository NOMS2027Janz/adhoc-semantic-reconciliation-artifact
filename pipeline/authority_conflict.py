#!/usr/bin/env python3
"""E9-conflict -- when two authority sources DISAGREE, does the agent surface it? (Brad pt 13, ext.)

E9 showed authority can be derived from an external artefact. This extension asks the harder,
honest-failure-mode question: when two sources CONTRADICT each other about who owns a field, does
the agent surface the conflict, or silently pick a side? The `authority_source_conflicting.json`
artefact gives two documents -- the interconnect agreement and a transport-side runbook -- that
disagree on the committed-rate owner (agreement: IP/Cascade; runbook: transport/Meridian). The
runbook reinforces the very transport lure a weak model already tends to follow, so silently
picking transport is the natural failure. Correct behaviour: authority='conflict' on committed-rate.

Two distinctions are tested:
  * committed-rate -- gold 'conflict' (sources contradict). Does the agent FLAG it, or silently
    pick X (the lure) or Y?
  * demarcation -- gold 'shared' (co-owned per ONE consistent statement). Must NOT be confused with
    a source conflict.

The other three fields are single-sourced (gold X/X/Y). Across the capability ladder.

    python pipeline/authority_conflict.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/authority_conflict.py --smoke

Writes results/authority_conflict.csv (one row per model x trial).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import SemanticModel   # noqa: E402
from run import load_dotenv                  # noqa: E402

CASE = "config_cross_domain"
COLS = ["case", "model", "trial", "authority_accuracy", "n_fields",
        "committed_verdict", "conflict_flagged", "committed_to_transport",
        "demarcation_correct", "demarcation_as_conflict", "false_conflicts"]

# gold for the conflicting-source case
GOLD = {"latency-realized": "X", "path-protection": "X", "service-class": "Y",
        "committed-rate": "conflict", "demarcation": "shared"}
SINGLE_SOURCE_FIELDS = ["latency-realized", "path-protection", "service-class", "demarcation"]

SYSTEM = (
    "Two systems from different domains meet at one seam: a Cascade (IP/VPN) service (Y) rides a "
    "Meridian (transport) circuit (X). For each shared field, decide the AUTHORITATIVE SOURCE OF "
    "TRUTH from the interconnect documents provided:\n"
    " - \"X\" if the transport realm owns / realises / measures it;\n"
    " - \"Y\" if the IP realm owns / sets / polices it;\n"
    " - \"shared\" if ONE consistent statement says it is co-owned by both;\n"
    " - \"conflict\" if the SOURCES DISAGREE with each other about who owns it.\n"
    "Distinguish carefully: 'shared' is a single co-ownership statement; 'conflict' is two sources "
    "that contradict each other. Carrying a value is not owning it. Do not silently pick a side when "
    "the documents disagree -- report 'conflict'. Return an authority for every field."
)


class _Attr(BaseModel):
    field_id: str
    authority: str


class _Result(BaseModel):
    attributions: list[_Attr]


def _concept_view(model_by_id, cid):
    c = model_by_id.get(cid)
    if not c:
        return {"id": cid}
    return {"id": cid, "label": c.label, "gloss": (c.gloss or "")}


def normalise(a: str) -> str:
    s = (a or "").strip().lower()
    return {"x": "X", "y": "Y", "shared": "shared", "both": "shared",
            "conflict": "conflict", "contested": "conflict", "disagree": "conflict"}.get(s, a)


def attribute(payload, model, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    from reconcile.compat import parse_compat
    completion = parse_compat(
        client, model,
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": json.dumps(payload, indent=2)}],
        _Result,
    )
    parsed = completion.choices[0].message.parsed
    return {a.field_id: normalise(a.authority) for a in (parsed.attributions if parsed else [])}


def score(attr: dict) -> dict:
    n = len(GOLD)
    correct = sum(1 for fid, g in GOLD.items() if attr.get(fid) == g)
    cv = attr.get("committed-rate")
    dem = attr.get("demarcation")
    # false conflicts: a single-source field wrongly reported as a conflict
    false_conf = sum(1 for f in SINGLE_SOURCE_FIELDS if attr.get(f) == "conflict")
    return {
        "authority_accuracy": round(correct / n, 3),
        "n_fields": n,
        "committed_verdict": cv,
        "conflict_flagged": int(cv == "conflict"),
        "committed_to_transport": int(cv == "X"),
        "demarcation_correct": int(dem == "shared"),
        "demarcation_as_conflict": int(dem == "conflict"),
        "false_conflicts": false_conf,
    }


def run_key(r):
    return (str(r["model"]), str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cdir = ROOT / "benchmark" / "cases" / CASE
    fields = json.loads((cdir / "pragmatics.json").read_text())["fields"]
    src = json.loads((cdir / "authority_source_conflicting.json").read_text())
    a = SemanticModel.from_json(cdir / "model_a_meridian.json")
    b = SemanticModel.from_json(cdir / "model_b_cascade.json")
    model_by_id = {c.id: c for c in list(a.concepts) + list(b.concepts)}

    seam = [{"field_id": f["id"],
             "transport_side_concept": _concept_view(model_by_id, f.get("pins", {}).get("a")),
             "ip_side_concept": _concept_view(model_by_id, f.get("pins", {}).get("b"))}
            for f in fields]
    payload = {"seam_fields": seam, "sources": src["sources"], "interconnect_documents": src["clauses"]}

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / "authority_conflict.csv"
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

    for model in models:
        for t in range(trials):
            if (str(model), str(t)) in done:
                continue
            print(f"[conflict] {model}/trial {t+1}", file=sys.stderr, flush=True)
            try:
                attr = attribute(payload, model)
            except Exception as e:  # noqa: BLE001
                print(f"      ! {e}", file=sys.stderr, flush=True)
                continue
            sc = score(attr)
            emit({"case": CASE, "model": model, "trial": t, **sc})
            print(f"      acc {sc['authority_accuracy']}  committed='{sc['committed_verdict']}' "
                  f"(flagged {sc['conflict_flagged']}, to-transport {sc['committed_to_transport']})  "
                  f"demarc-ok {sc['demarcation_correct']}  false-conflicts {sc['false_conflicts']}",
                  file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    if not rows:
        return
    models = []
    for r in rows:
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

    print("\n=== E9-conflict  Do agents surface a source conflict, or silently pick a side? ===")
    print(f"    {'model':13}{'auth acc':10}{'conflict flagged':18}{'silently->transport':21}{'demarc ok':11}{'false conflicts':15}")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        print(f"    {m:13}{mean(rs,'authority_accuracy'):<10.2f}{mean(rs,'conflict_flagged'):<18.2f}"
              f"{mean(rs,'committed_to_transport'):<21.2f}{mean(rs,'demarcation_correct'):<11.2f}"
              f"{mean(rs,'false_conflicts'):<15.2f}")
    print("\n    Reading: 'conflict flagged' is the honest response on committed-rate (the two sources")
    print("    disagree); 'silently->transport' is the failure of following the lure without noticing")
    print("    the contradiction; 'demarc ok' checks co-ownership isn't confused with conflict; 'false")
    print("    conflicts' flags over-calling conflict on the single-sourced fields.")


if __name__ == "__main__":
    raise SystemExit(main())
