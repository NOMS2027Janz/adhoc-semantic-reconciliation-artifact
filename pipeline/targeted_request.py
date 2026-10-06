#!/usr/bin/env python3
"""E12 -- recognising under-specification and issuing a TARGETED request (Brad, one-off scenario).

A candidate correspondence can be under-determined by the normal exchange: the surface (a matching
label + kind) is not enough to tell a true correspondence from a false cognate, but ONE extra
attribute, available on request, resolves it. The question is whether the agent RECOGNISES the
shortfall and asks for the RIGHT thing -- a targeted request -- rather than guessing from the surface
or asking for everything (the decisive-experiment reflex). Case: `underspecified_pairs`; each item
has a menu of requestable attributes, exactly one of which disambiguates.

Two arms per (model, item, trial):
  * surface_only -- decide from label + kind alone (no requests). The accuracy ceiling of guessing.
  * request      -- triage: is the surface sufficient? if not, request specific attributes; then a
                    resolve step is given only the requested values and decides. Scored on whether
                    the agent asked (recognised the shortfall), whether it asked for the
                    disambiguating attribute (hit), how many it asked for (targeting), and final
                    correctness.

    python pipeline/targeted_request.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 3
    python pipeline/targeted_request.py --smoke

Writes results/targeted_request.csv (one row per model x item x arm x trial).
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
from run import load_dotenv   # noqa: E402

CASE = "underspecified_pairs"
COLS = ["case", "model", "item", "arm", "trial", "asked", "hit", "n_requested",
        "targeted", "final_correct", "gold_corresponds"]

TRIAGE_SYSTEM = (
    "You must decide whether concept A and concept B denote the SAME thing (a correspondence) or "
    "are merely a look-alike (a false cognate). You are shown only their SURFACE: label, kind, "
    "system. The surface may be insufficient -- a matching label does not settle it. You may REQUEST "
    "the value of specific attributes (from the given menu) for both sides, but each request has a "
    "cost, so ask for the SMALLEST set that would actually resolve the question. If the surface "
    "alone is enough, set sufficient=true and give your decision; otherwise set sufficient=false and "
    "list only the attribute names you need."
)
RESOLVE_SYSTEM = (
    "You are deciding whether concept A and concept B denote the SAME thing. You now have the values "
    "of the attributes you requested, for both sides. Decide: do they correspond?"
)


class Triage(BaseModel):
    sufficient: bool
    corresponds_if_sufficient: bool
    requests: list[str]


class Resolve(BaseModel):
    corresponds: bool


def _client(client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    return client


def _parse(client, model, system, user, schema):
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(user, indent=2)}],
        response_format=schema,
    )
    return completion.choices[0].message.parsed


def surface_view(it):
    return {"A": it["a"], "B": it["b"]}


def run_surface_only(it, model, client):
    user = {**surface_view(it),
            "instruction": "Decide from the surface alone. Set sufficient=true and give corresponds_if_sufficient."}
    t = _parse(client, model, TRIAGE_SYSTEM, user, Triage)
    decision = bool(t.corresponds_if_sufficient) if t else False
    return {"asked": 0, "hit": 0, "n_requested": 0, "targeted": 0,
            "final_correct": int(decision == it["gold_corresponds"])}


def run_request(it, model, client):
    menu = it["requestable"]
    triage_user = {**surface_view(it), "requestable_attributes": menu}
    t = _parse(client, model, TRIAGE_SYSTEM, triage_user, Triage)
    if t is None:
        return {"asked": 0, "hit": 0, "n_requested": 0, "targeted": 0, "final_correct": 0}

    if t.sufficient:
        decision = bool(t.corresponds_if_sufficient)
        return {"asked": 0, "hit": 0, "n_requested": 0, "targeted": 0,
                "final_correct": int(decision == it["gold_corresponds"])}

    # agent asked: keep only valid menu attributes
    reqs = [r for r in t.requests if r in menu]
    hit = int(it["disambiguating"] in reqs)
    revealed = {r: {"A": it["facts"]["a"].get(r), "B": it["facts"]["b"].get(r)} for r in reqs}
    resolve_user = {**surface_view(it), "requested_attributes": revealed}
    r = _parse(client, model, RESOLVE_SYSTEM, resolve_user, Resolve)
    decision = bool(r.corresponds) if r else False
    return {"asked": 1, "hit": hit, "n_requested": len(reqs),
            "targeted": int(hit and len(reqs) == 1),
            "final_correct": int(decision == it["gold_corresponds"])}


def run_key(row):
    return (str(row["model"]), row["item"], row["arm"], str(row["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    items = json.loads((ROOT / "benchmark" / "cases" / CASE / "items.json").read_text())["items"]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials, items = models[:1], 1, items[:2]
    trials = max(1, args.trials)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    client = _client()

    out = ROOT / "results" / "targeted_request.csv"
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

    arms = {"surface_only": run_surface_only, "request": run_request}
    total = len(models) * len(items) * len(arms) * trials
    i = 0
    for model in models:
        for it in items:
            for arm, fn in arms.items():
                for t in range(trials):
                    i += 1
                    key = (str(model), it["id"], arm, str(t))
                    if key in done:
                        continue
                    print(f"[{i}/{total}] {model}/{it['id']}/{arm}/t{t+1}", file=sys.stderr, flush=True)
                    try:
                        res = fn(it, model, client)
                    except Exception as e:  # noqa: BLE001
                        print(f"      ! {e}", file=sys.stderr, flush=True)
                        continue
                    emit({"case": CASE, "model": model, "item": it["id"], "arm": arm, "trial": t,
                          "gold_corresponds": it["gold_corresponds"], **res})

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

    print("\n=== E12  Recognising under-specification and issuing a targeted request ===")
    print(f"    {'model':13}{'surface acc':13}{'request acc':13}{'ask rate':10}{'hit rate':10}{'targeted':10}{'avg #req':9}")
    for m in models:
        so = [r for r in rows if r["model"] == m and r["arm"] == "surface_only"]
        rq = [r for r in rows if r["model"] == m and r["arm"] == "request"]
        asked = [r for r in rq if str(r["asked"]) == "1"]
        def s(v):
            return f"{v:.2f}" if v is not None else "-"
        print(f"    {m:13}{s(mean(so,'final_correct')):13}{s(mean(rq,'final_correct')):13}"
              f"{s(mean(rq,'asked')):10}{s(mean(rq,'hit')):10}{s(mean(rq,'targeted')):10}"
              f"{s(mean(asked,'n_requested')):9}")
    print("\n    Reading: 'request acc' >> 'surface acc' means asking is necessary and the agent uses")
    print("    it; 'ask rate' is recognition of the shortfall; 'hit' is asking for the disambiguating")
    print("    attribute; 'targeted' is asking for ONLY it (vs the ask-for-everything reflex).")


if __name__ == "__main__":
    raise SystemExit(main())
