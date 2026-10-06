#!/usr/bin/env python3
"""E9 -- can cross-domain authority be DERIVED from an external artefact? (Brad pt 13.)

Which realm governs a shared seam field (transport X, IP Y, or co-owned 'shared') is hand-authored
in the study. Brad asks whether it can instead be derived from an external artefact -- an
interconnect agreement / SLA / system-of-record -- and whether the agent correctly DEFERS on a
genuinely contested field rather than inventing an owner. E9 gives the agent that artefact
(`authority_source.json`) and compares two arms across the capability ladder:

  * STRUCTURE (baseline): attribute each field to X / Y / shared from the concept structure alone
    (label + gloss of the pinned concepts on each side), as the study does today.
  * ARTEFACT: the same, but also given the interconnect-agreement clauses, from which authority is
    to be derived.

Two fields are diagnostic:
  * committed-rate -- gold authority Y (an IP-owned SLA commitment), but transport CARRIES it, so a
    naive attributor gives it to transport ('transport claims everything it carries'). The artefact
    states carriage != ownership; does that correct the lure?
  * demarcation -- gold 'shared' (co-owned). Does the agent DEFER, or invent a single owner? Does
    the artefact help the deferral or tempt a pick?

    python pipeline/authority_derivation.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/authority_derivation.py --smoke

Writes results/authority_derivation.csv (one row per model x arm x trial).
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
COLS = ["case", "model", "arm", "trial", "authority_accuracy", "n_fields",
        "committed_rate_correct", "demarcation_correct", "contested_to_transport",
        "reasoning_tokens"]

SYSTEM = (
    "Two bespoke systems from DIFFERENT domains meet at one seam: a Cascade (IP/VPN) service (call "
    "it Y) rides a Meridian (transport/optical) circuit (call it X). For each shared requirement "
    "FIELD at the seam, decide which realm is the AUTHORITATIVE SOURCE OF TRUTH for the field's "
    "value:\n"
    " - answer \"X\" if the transport realm owns / realises / measures it;\n"
    " - answer \"Y\" if the IP realm owns / sets / polices it;\n"
    " - answer \"shared\" if it is a single co-owned point that neither side governs alone.\n"
    "Carrying a value is not the same as owning it: a realm that merely transports another realm's "
    "commitment does not gain authority over it. Return an authority for every field."
)
ARTEFACT_NOTE = (
    "\n\nYou are also given the INTERCONNECT AGREEMENT: numbered clauses stating, in operational "
    "terms, who owns, realises or measures each field. Derive each field's authority from the "
    "relevant clause. A clause that says a field is jointly defined / co-owned / requires bilateral "
    "sign-off means the authority is \"shared\"."
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


def build_payload(fields, model_by_id, clauses=None):
    seam = []
    for f in fields:
        pins = f.get("pins", {})
        seam.append({
            "field_id": f["id"],
            "transport_side_concept": _concept_view(model_by_id, pins.get("a")),
            "ip_side_concept": _concept_view(model_by_id, pins.get("b")),
        })
    payload = {"seam_fields": seam}
    if clauses is not None:
        payload["interconnect_agreement"] = clauses
    return payload


def normalise(auth: str) -> str:
    a = (auth or "").strip().lower()
    return {"x": "X", "y": "Y", "shared": "shared", "both": "shared"}.get(a, auth)


def attribute(payload, system, model, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    from reconcile.compat import parse_compat
    completion = parse_compat(
        client, model,
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(payload, indent=2)}],
        _Result,
    )
    parsed = completion.choices[0].message.parsed
    out = {}
    for a in (parsed.attributions if parsed else []):
        out[a.field_id] = normalise(a.authority)
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    rtok = getattr(details, "reasoning_tokens", None) if details else None
    return out, rtok


def score(attr: dict, gold: dict) -> dict:
    n = len(gold)
    correct = sum(1 for fid, g in gold.items() if attr.get(fid) == g)
    return {
        "authority_accuracy": round(correct / n, 3) if n else 0.0,
        "n_fields": n,
        "committed_rate_correct": int(attr.get("committed-rate") == gold.get("committed-rate")),
        "demarcation_correct": int(attr.get("demarcation") == gold.get("demarcation")),
        "contested_to_transport": int(attr.get("committed-rate") == "X"),
    }


def run_key(r):
    return (str(r["model"]), r["arm"], str(r["trial"]))


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
    gold = {f["id"]: f["authority"] for f in fields}
    clauses = json.loads((cdir / "authority_source.json").read_text())["clauses"]
    a = SemanticModel.from_json(cdir / "model_a_meridian.json")
    b = SemanticModel.from_json(cdir / "model_b_cascade.json")
    model_by_id = {c.id: c for c in list(a.concepts) + list(b.concepts)}

    payloads = {
        "structure": (build_payload(fields, model_by_id, clauses=None), SYSTEM),
        "artefact": (build_payload(fields, model_by_id, clauses=clauses), SYSTEM + ARTEFACT_NOTE),
    }

    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / "authority_derivation.csv"
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
        for arm, (payload, system) in payloads.items():
            for t in range(trials):
                key = (str(model), arm, str(t))
                if key in done:
                    continue
                print(f"[{arm}] {model}/trial {t+1}", file=sys.stderr, flush=True)
                try:
                    attr, rtok = attribute(payload, system, model)
                except Exception as e:  # noqa: BLE001
                    print(f"      ! {e}", file=sys.stderr, flush=True)
                    continue
                sc = score(attr, gold)
                emit({"case": CASE, "model": model, "arm": arm, "trial": t,
                      "reasoning_tokens": rtok if rtok is not None else "", **sc})
                print(f"      acc {sc['authority_accuracy']}  committed-rate ok {sc['committed_rate_correct']} "
                      f"(to transport {sc['contested_to_transport']})  demarcation-deferred {sc['demarcation_correct']}",
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

    print("\n=== E9  Authority derivation: structure-only vs given the interconnect artefact ===")
    print(f"    {'model':13}{'arm':11}{'auth acc':10}{'committed-rate ok':19}{'demarcation deferred':20}")
    for m in models:
        for arm in ("structure", "artefact"):
            rs = [r for r in rows if r["model"] == m and r["arm"] == arm]
            if not rs:
                continue
            print(f"    {m:13}{arm:11}{mean(rs,'authority_accuracy'):<10.2f}"
                  f"{mean(rs,'committed_rate_correct'):<19.2f}{mean(rs,'demarcation_correct'):<20.2f}")
    print("\n    Reading: does the external artefact raise authority accuracy over structure-only")
    print("    inference (esp. correcting the committed-rate lure toward transport), and does the")
    print("    agent DEFER on the co-owned demarcation ('shared') rather than invent an owner --")
    print("    and are both capability-gated?")


if __name__ == "__main__":
    raise SystemExit(main())
