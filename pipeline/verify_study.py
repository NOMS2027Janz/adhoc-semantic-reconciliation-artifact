#!/usr/bin/env python3
"""Driver for the verification study: mode x placement x model over the seeded verify set.

For every proposed correspondence, three verifier modes render a verdict, and we score the
verifier's own quality — does it PASS what is correct and FAIL what is wrong — plus its reach
across the cognition spectrum:

  * byte_round_trip and virtual_operation are deterministic (byte is placement-independent;
    virtual uses the instance Oracle and so is placement-gated), scored once per placement.
  * invariant_round_trip calls the model, scored per model x placement x trial.

Metrics per (mode, placement[, model, trial]): verification precision (of what it passed, the
fraction truly correct), catch rate (of the wrong, the fraction failed), false-pass rate, reach
(the fraction it could decide), and the catch rate split by category (meaning-visible vs
byte-clean). Correctness is the currency; tokens are effort. Checkpointed, en-route capture.

  python verify_study.py --smoke                      # one model, deterministic + invariant
  python verify_study.py --model gpt-5.6-sol --trials 6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import json                                                       # noqa: E402
from reconcile.instance import load_instance_case, Oracle         # noqa: E402
from reconcile import verify_modes as VM                          # noqa: E402
from run import load_dotenv                                       # noqa: E402

PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
COLUMNS = ["case", "mode", "model", "placement", "trial",
           "n_correct", "n_wrong", "pass_correct", "fail_correct", "refer_correct",
           "catch_wrong", "false_pass", "refer_wrong",
           "verification_precision", "catch_rate", "false_pass_rate", "reach",
           "mv_catch_rate", "bc_catch_rate", "total_tokens", "reasoning_tokens", "latency_s"]


def load_verify(case_dir):
    p = Path(case_dir)
    proposals = json.loads((p / "proposals.json").read_text())
    gold = json.loads((p / "verify_gold.json").read_text())["verdicts"]
    return proposals, gold


def score(verdicts, proposals, gold, effort=None) -> dict:
    n_correct = n_wrong = 0
    pass_correct = fail_correct = refer_correct = 0
    catch_wrong = false_pass = refer_wrong = 0
    mv_total = mv_catch = bc_total = bc_catch = 0
    for p in proposals["proposals"]:
        g = gold[p["id"]]
        v = verdicts.get(p["id"], VM.REFER)
        if g["truth_pass"]:
            n_correct += 1
            pass_correct += (v == VM.PASS)
            fail_correct += (v == VM.FAIL)
            refer_correct += (v == VM.REFER)
        else:
            n_wrong += 1
            catch_wrong += (v == VM.FAIL)
            false_pass += (v == VM.PASS)
            refer_wrong += (v == VM.REFER)
            if g["category"] == "meaning-visible":
                mv_total += 1; mv_catch += (v == VM.FAIL)
            elif g["category"] == "byte-clean":
                bc_total += 1; bc_catch += (v == VM.FAIL)
    decided = sum(1 for p in proposals["proposals"]
                  if verdicts.get(p["id"], VM.REFER) != VM.REFER)
    passed = pass_correct + false_pass
    row = {
        "n_correct": n_correct, "n_wrong": n_wrong,
        "pass_correct": pass_correct, "fail_correct": fail_correct, "refer_correct": refer_correct,
        "catch_wrong": catch_wrong, "false_pass": false_pass, "refer_wrong": refer_wrong,
        "verification_precision": round(pass_correct / passed, 3) if passed else "",
        "catch_rate": round(catch_wrong / n_wrong, 3) if n_wrong else "",
        "false_pass_rate": round(false_pass / n_wrong, 3) if n_wrong else "",
        "reach": round(decided / len(proposals["proposals"]), 3),
        "mv_catch_rate": round(mv_catch / mv_total, 3) if mv_total else "",
        "bc_catch_rate": round(bc_catch / bc_total, 3) if bc_total else "",
        "total_tokens": (effort or {}).get("total_tokens", ""),
        "reasoning_tokens": (effort or {}).get("reasoning_tokens", ""),
        "latency_s": (effort or {}).get("latency_s", ""),
    }
    return row


def deterministic_verdicts(mode, proposals, placement, case):
    if mode == "byte":
        return {p["id"]: VM.byte_round_trip(p) for p in proposals["proposals"]}
    if mode == "virtual":
        oracle = Oracle(case, placement, budget=None)  # unbounded; deterministic
        return {p["id"]: VM.virtual_operation(p, oracle) for p in proposals["proposals"]}
    raise ValueError(mode)


def run_key(r):
    return (r["mode"], str(r["model"]), r["placement"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="one model, one trial")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    proposals, gold = load_verify(ROOT / "benchmark" / "cases" / "verify_hard")
    case = load_instance_case(ROOT / "benchmark" / "cases" / "instance_hard")
    inv = proposals.get("invariants", [])
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    if args.smoke:
        models, args.trials = models[:1], 1
    trials = max(1, args.trials)

    need_api = True  # invariant_round_trip needs the model
    load_dotenv()
    if need_api and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see SETUP_OPENAI.md", file=sys.stderr)
        return 2

    suffix = f"_{models[0]}" if len(models) == 1 else ""
    out = ROOT / "results" / f"verify_verify_hard{suffix}{'_smoke' if args.smoke else ''}.csv"
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
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader(); fh.flush(); rows, done = [], set()

    def emit(base):
        rows.append(base); done.add(run_key(base))
        if writer is not None:
            writer.writerow(base); fh.flush()

    # deterministic modes: once per placement (model-independent)
    for pl in PLACEMENTS:
        for mode in ("byte", "virtual"):
            key = (mode, "deterministic", pl, "0")
            if key in done:
                continue
            verdicts = deterministic_verdicts(mode, proposals, pl, case)
            base = {"case": "verify_hard", "mode": mode, "model": "deterministic",
                    "placement": pl, "trial": 0, **score(verdicts, proposals, gold)}
            emit(base)
            print(f"[det] {mode}/{pl}: catch {base['catch_rate']} false-pass {base['false_pass_rate']} "
                  f"reach {base['reach']} (mv {base['mv_catch_rate']} bc {base['bc_catch_rate']})",
                  file=sys.stderr, flush=True)

    # invariant_round_trip: model x placement x trial
    total = len(models) * len(PLACEMENTS) * trials
    i = 0
    for model in models:
        for pl in PLACEMENTS:
            for t in range(trials):
                i += 1
                key = ("invariant", str(model), pl, str(t))
                if key in done:
                    print(f"[{i}/{total}] invariant {model}/{pl}/{t} (skip)", file=sys.stderr)
                    continue
                print(f"[{i}/{total}] invariant {model}/{pl}/trial {t+1}", file=sys.stderr, flush=True)
                try:
                    verdicts, effort = VM.invariant_round_trip(
                        proposals["proposals"], pl, model, inv)
                except Exception as e:
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                base = {"case": "verify_hard", "mode": "invariant", "model": model,
                        "placement": pl, "trial": t, **score(verdicts, proposals, gold, effort)}
                emit(base)
                print(f"      catch {base['catch_rate']} false-pass {base['false_pass_rate']} "
                      f"prec {base['verification_precision']} (mv {base['mv_catch_rate']} bc {base['bc_catch_rate']})",
                      file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarize(rows):
    if not rows:
        return

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def m(rs, k):
        xs = [fl(r[k]) for r in rs if fl(r[k]) is not None]
        return sum(xs) / len(xs) if xs else 0.0
    print("\n=== catch rate / false-pass / reach by mode x placement ===")
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    for mode in modes:
        print(f"\n-- {mode} --")
        for pl in PLACEMENTS:
            sel = [r for r in rows if r["mode"] == mode and r["placement"] == pl]
            if sel:
                print(f"  {pl:15s} catch {m(sel,'catch_rate'):.2f}  false-pass {m(sel,'false_pass_rate'):.2f}  "
                      f"reach {m(sel,'reach'):.2f}  mv {m(sel,'mv_catch_rate'):.2f}  bc {m(sel,'bc_catch_rate'):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
