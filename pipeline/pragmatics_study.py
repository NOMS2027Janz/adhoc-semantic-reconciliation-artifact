#!/usr/bin/env python3
"""Pragmatics wave for the cross-domain setting: realm and authority as a scored axis.

    python pragmatics_study.py --case config_cross_domain

For each shared requirement field at the cross-domain seam, the agent must attribute the
AUTHORITATIVE SOURCE OF TRUTH — X (transport / Meridian), Y (IP / Cascade), or shared — and
is scored against the pragmatics gold (benchmark/cases/<case>/pragmatics.json). Sweeps the
cognition spectrum x reference-on/off x the model ladder x trials, writing its own segmented
CSV (results/pragmatics_<case>.csv), resumable. Correspondence is settled elsewhere; this
measures only the pragmatic layer the first two settings held fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Case                                     # noqa: E402
from reconcile.stacks.agent_pragmatics import PragmaticsStack        # noqa: E402
from run import load_dotenv                                          # noqa: E402

PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
COLUMNS = ["case", "model", "placement", "reference", "trial", "submitted", "n_fields",
           "authority_correct", "authority_accuracy", "over_transport", "over_ip",
           "shared_missed", "fields", "reasoning_tokens", "latency_s"]


def score(rec, fields) -> dict:
    gold = {f["id"]: f["authority"] for f in fields}
    correct = over_x = over_y = shared_missed = 0
    marks = []
    for f in fields:
        fid = f["id"]
        want = gold[fid]
        got = rec.attributions.get(fid, "?")
        ok = (got == want)
        correct += int(ok)
        if got == "X" and want != "X":
            over_x += 1
        if got == "Y" and want != "Y":
            over_y += 1
        if want == "shared" and got != "shared":
            shared_missed += 1
        marks.append(f"{fid}:{got}{'ok' if ok else '!='+want}")
    n = len(fields)
    return {"n_fields": n, "authority_correct": correct,
            "authority_accuracy": round(correct / n, 3) if n else 0.0,
            "over_transport": over_x, "over_ip": over_y, "shared_missed": shared_missed,
            "fields": ";".join(marks)}


def run_key(r):
    return (str(r["model"]), r["placement"], str(r["reference"]), str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="config_cross_domain")
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--placements", default="both_cognitive,one_inert,both_inert")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (default: resume)")
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2

    case_dir = ROOT / "benchmark" / "cases" / args.case
    case = Case.load(case_dir)
    pfile = case_dir / "pragmatics.json"
    if not pfile.exists():
        print(f"no pragmatics.json in {args.case}; nothing to run", file=sys.stderr)
        return 2
    fields = json.loads(pfile.read_text())["fields"]
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    trials = max(1, args.trials)
    write = not args.no_write

    suffix = f"_{models[0]}" if len(models) == 1 else ""
    out = ROOT / "results" / f"pragmatics_{args.case}{suffix}.csv"
    rows, done = [], set()
    if write:
        out.parent.mkdir(exist_ok=True)
        if out.exists() and not args.fresh:
            with out.open(newline="") as f:
                rows = list(csv.DictReader(f))
            done = {run_key(r) for r in rows}
            print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)
    fh = writer = None
    if write:
        fresh_file = args.fresh or not out.exists()
        fh = out.open("w" if fresh_file else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh_file:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    plan = [(m, pl, ref) for m in models for pl in placements for ref in (False, True)]
    total = len(plan) * trials
    print(f"Pragmatics study: {len(plan)} conditions x {trials} trials = {total} runs, "
          f"case {args.case}, {len(fields)} seam fields", file=sys.stderr)

    i = 0
    for (m, pl, use_ref) in plan:
        for t in range(trials):
            i += 1
            key = (m, pl, str(use_ref), str(t))
            if key in done:
                print(f"[{i}/{total}] {m}/{pl}/{'ref' if use_ref else 'no-ref'}/t{t} (skip)",
                      file=sys.stderr, flush=True)
                continue
            print(f"[{i}/{total}] {m}/{pl}/{'ref' if use_ref else 'no-ref'}/trial {t+1}",
                  file=sys.stderr, flush=True)
            stack = PragmaticsStack(case, fields, use_reference=use_ref, model=m)
            try:
                rec = stack.reconcile(placement=pl)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            s = score(rec, fields)
            row = {"case": args.case, "model": m, "placement": pl,
                   "reference": "ref" if use_ref else "no-ref", "trial": t,
                   "submitted": rec.submitted,
                   "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
                   "latency_s": rec.effort.get("latency_s", ""), **s}
            rows.append(row); done.add(key)
            if writer is not None:
                writer.writerow(row); fh.flush()
            print(f"      auth-acc {s['authority_accuracy']} ({s['authority_correct']}/{s['n_fields']}) "
                  f"overX {s['over_transport']} overY {s['over_ip']}  [{s['fields']}]",
                  file=sys.stderr, flush=True)
    if fh is not None:
        fh.close()

    # brief summary
    def mean(rs, k):
        xs = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
        return sum(xs) / len(xs) if xs else 0.0
    print("\nAuthority accuracy by model x placement x reference:", file=sys.stderr)
    for m in models:
        for pl in placements:
            for ref in ("no-ref", "ref"):
                sel = [r for r in rows if r["model"] == m and r["placement"] == pl and r["reference"] == ref]
                if sel:
                    print(f"  {m:12s} {pl:15s} {ref:7s} acc {mean(sel,'authority_accuracy'):.2f}",
                          file=sys.stderr)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
