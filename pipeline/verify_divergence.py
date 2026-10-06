#!/usr/bin/env python3
"""E6 -- how distinct are the negotiation and the decisive experiment? (Brad review, pt 9).

The study has two things that verify a correspondence, and in the main runs both are ultimately
scored against the SAME answer key, so Brad asks how distinct they really are:

  * the AGENT's own verdict -- `invariant_round_trip`, an LLM reading the two sides' static
    records and judging whether the correspondence preserves the semantic invariants. This is the
    "negotiation / assertion": it reaches every proposal but is capability-gated.
  * the DECISIVE EXPERIMENT -- `virtual_operation`, which *exercises* the correspondence on the
    graph (provision-and-read-back for services; interrogate an authoritative fact on both sides
    for devices/sections). It consults NO gold; it is an independent oracle. But it can only act
    where a side is live, so its reach recedes as cognition recedes.

This driver runs BOTH on the same proposals and measures their **mutual divergence, without gold**:
where both decide, do they agree? when they disagree, which way (agent PASSes what the experiment
FAILs -- an over-confident assertion -- or the reverse)? Gold is used ONLY afterwards, to adjudicate
the disagreements (when they diverge, is the independent experiment the better authority?) and to
report each one's standalone accuracy -- so the reader can see the two are distinct authorities, not
one grounded in the other. The reach gap (where the experiment refers and the agent must stand
alone) is itself a headline: it is exactly where "the agents verify rather than assert" has to hold
on the agent's word, because the decisive experiment cannot run.

    python verify_divergence.py --smoke                          # one model, one trial
    python verify_divergence.py --model gpt-5.6-sol,gpt-5-mini,gpt-5-nano --trials 6

Writes results/verify_divergence_verify_hard.csv (one row per model x placement x trial), plus a
deterministic byte-vs-oracle reference row per placement.
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
from reconcile.instance import load_instance_case, Oracle   # noqa: E402
from reconcile import verify_modes as VM                    # noqa: E402
from run import load_dotenv                                 # noqa: E402

PLACEMENTS = ["both_cognitive", "one_inert", "both_inert"]
COLUMNS = ["case", "agent_source", "model", "placement", "trial",
           "n_proposals", "oracle_reach", "agent_reach", "co_decided",
           "agree", "disagree", "agreement_rate",
           "agent_pass_oracle_fail", "agent_fail_oracle_pass",
           "agent_only_decided", "oracle_only_decided",
           "disagree_oracle_right", "disagree_agent_right",
           "agent_acc_where_decided", "oracle_acc_where_decided"]


def load_verify(case_dir: Path):
    proposals = json.loads((case_dir / "proposals.json").read_text())
    gold = json.loads((case_dir / "verify_gold.json").read_text())["verdicts"]
    return proposals, gold


def oracle_verdicts(proposals, placement, case) -> dict:
    """The independent decisive experiment: exercise each correspondence on the graph. No gold."""
    oracle = Oracle(case, placement, budget=None)   # unbounded; deterministic
    return {p["id"]: VM.virtual_operation(p, oracle) for p in proposals["proposals"]}


def byte_verdicts(proposals) -> dict:
    return {p["id"]: VM.byte_round_trip(p) for p in proposals["proposals"]}


def _acc_where_decided(verd: dict, gold: dict) -> str:
    # restrict to real proposal ids: a weak model can return a verdict for a fabricated id
    # that is not among the proposals (and so not in gold) -- ignore those here.
    dec = [pid for pid, v in verd.items() if v != VM.REFER and pid in gold]
    if not dec:
        return ""
    right = sum(1 for pid in dec if (verd[pid] == VM.PASS) == gold[pid]["truth_pass"])
    return round(right / len(dec), 3)


def compare(agent: dict, oracle: dict, proposals, gold) -> dict:
    """Divergence of an agent verdict-set against the independent oracle, WITHOUT gold; gold is
    consulted only to adjudicate the disagreements and report standalone accuracy."""
    ids = [p["id"] for p in proposals["proposals"]]
    agent_dec = {i for i in ids if agent.get(i, VM.REFER) != VM.REFER}
    oracle_dec = {i for i in ids if oracle.get(i, VM.REFER) != VM.REFER}
    co = agent_dec & oracle_dec

    agree = sum(1 for i in co if agent[i] == oracle[i])
    disagree = len(co) - agree
    apof = sum(1 for i in co if agent[i] == VM.PASS and oracle[i] == VM.FAIL)
    afop = sum(1 for i in co if agent[i] == VM.FAIL and oracle[i] == VM.PASS)

    # adjudicate the disagreements with gold: who was right?
    dis_ids = [i for i in co if agent[i] != oracle[i]]
    oracle_right = sum(1 for i in dis_ids if (oracle[i] == VM.PASS) == gold[i]["truth_pass"])
    agent_right = sum(1 for i in dis_ids if (agent[i] == VM.PASS) == gold[i]["truth_pass"])

    return {
        "n_proposals": len(ids),
        "oracle_reach": len(oracle_dec),
        "agent_reach": len(agent_dec),
        "co_decided": len(co),
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": round(agree / len(co), 3) if co else "",
        "agent_pass_oracle_fail": apof,
        "agent_fail_oracle_pass": afop,
        "agent_only_decided": len(agent_dec - oracle_dec),
        "oracle_only_decided": len(oracle_dec - agent_dec),
        "disagree_oracle_right": oracle_right,
        "disagree_agent_right": agent_right,
        "agent_acc_where_decided": _acc_where_decided(agent, gold),
        "oracle_acc_where_decided": _acc_where_decided(oracle, gold),
    }


def run_key(r):
    return (r["agent_source"], str(r["model"]), r["placement"], str(r["trial"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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

    # the independent oracle is deterministic: compute once per placement, up front
    oracle_by_pl = {pl: oracle_verdicts(proposals, pl, case) for pl in PLACEMENTS}

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2

    out = ROOT / "results" / "verify_divergence_verify_hard.csv"
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

    # deterministic reference: the naive byte verifier vs the oracle (a floor for divergence)
    for pl in PLACEMENTS:
        key = ("byte", "deterministic", pl, "0")
        if key not in done:
            base = {"case": "verify_hard", "agent_source": "byte", "model": "deterministic",
                    "placement": pl, "trial": 0,
                    **compare(byte_verdicts(proposals), oracle_by_pl[pl], proposals, gold)}
            emit(base)
            print(f"[det] byte vs oracle / {pl}: co={base['co_decided']} agree={base['agree']} "
                  f"agent_pass_oracle_fail={base['agent_pass_oracle_fail']}", file=sys.stderr, flush=True)

    # the agent verifier vs the oracle: model x placement x trial
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
                    agent, _effort = VM.invariant_round_trip(proposals["proposals"], pl, model, inv)
                except Exception as e:  # noqa: BLE001
                    print(f"      ! error: {e}", file=sys.stderr, flush=True)
                    continue
                base = {"case": "verify_hard", "agent_source": "invariant", "model": model,
                        "placement": pl, "trial": t,
                        **compare(agent, oracle_by_pl[pl], proposals, gold)}
                emit(base)
                print(f"      co={base['co_decided']} agree={base['agree']} disagree={base['disagree']} "
                      f"(agent_pass_oracle_fail={base['agent_pass_oracle_fail']}, "
                      f"oracle_right_when_split={base['disagree_oracle_right']}/{base['disagree']}), "
                      f"agent_only={base['agent_only_decided']}", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarise(rows)
    if write:
        print(f"\nWrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


def summarise(rows):
    inv = [r for r in rows if r["agent_source"] == "invariant"]
    if not inv:
        return

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def mean(rs, k):
        xs = [fl(r[k]) for r in rs if fl(r[k]) is not None]
        return sum(xs) / len(xs) if xs else None

    models = []
    for r in inv:
        if r["model"] not in models:
            models.append(r["model"])
    print("\n=== Agent verdict vs the independent decisive experiment (no gold) ===")
    print("    where BOTH decide: agreement; and the reach gap where the experiment cannot run")
    hdr = f"    {'model':13}{'placement':16}{'co-dec':7}{'agree%':8}{'a.pass/o.fail':14}{'agent-only':11}"
    print(hdr)
    for model in models:
        for pl in PLACEMENTS:
            rs = [r for r in inv if r["model"] == model and r["placement"] == pl]
            if not rs:
                continue
            ar = mean(rs, "agreement_rate")
            co = mean(rs, "co_decided")
            apof = mean(rs, "agent_pass_oracle_fail")
            ao = mean(rs, "agent_only_decided")
            ar_s = f"{ar:.2f}" if ar is not None else "  -"
            print(f"    {model:13}{pl:16}{co:<7.1f}{ar_s:<8}{apof:<14.1f}{ao:<11.1f}")
    print("\n    Reading: where the experiment has reach (both_cognitive), agreement measures how")
    print("    distinct the two really are. Where its reach collapses (inert), 'agent-only' is the")
    print("    load the agent's assertion must carry alone -- the experiment cannot adjudicate there.")


if __name__ == "__main__":
    raise SystemExit(main())
