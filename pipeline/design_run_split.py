#!/usr/bin/env python3
"""E19 -- the design-time / run-time split, over the four semantic-model layers.

Section 13.9 draws the line the production question turns on: some of the meaning is observed and
frozen at lift time; some is judged live and must be resolved at run time. The semantic-model
architecture names four layers -- the schematic ontology (TBox: identities, kinds, relations), the
concrete instances (ABox), PROVENANCE (who asserted / realises / is the system of record for a
fact), and PRAGMATICS (what a thing is for, whose use governs it, whether it can be delivered now).
This experiment tests whether an agent can sort the facts a cross-domain reconciliation needs onto
those layers AND decide, for each, whether it freezes at design time or must be resolved live.

The provenance layer matters here for a reason the study half-missed: the "authority" of a seam
field bundles two things. WHO IS THE SOURCE OF RECORD for a field (who realises/measures/owns its
value) is PROVENANCE, and it FREEZES -- you can settle once who to ask. The CURRENT VALUE that source
holds is an INSTANCE fact that drifts, and is LIVE. "Whose use governs" and "can it be delivered now"
are PRAGMATICS, and are LIVE. So the clean statement of 13.9 is: freeze the schema and the
provenance (what a thing is, and who to ask); resolve live the instances and the pragmatics (what
they currently say, and what it is for now).

Gold layer -> timing: schematic FREEZE, provenance FREEZE, instance LIVE, pragmatic LIVE. The agent
returns, per fact, a layer and a timing with a confidence and reason. We score layer accuracy and
timing accuracy; two directed timing errors -- FREEZE-as-LIVE (over-caution) and, the dangerous one,
LIVE-as-FREEZE (a staleness hazard, the confident-error analog for this axis); and one diagnostic of
the conflation the study itself made -- PROVENANCE-as-PRAGMATIC (treating who-is-the-source as a live
governance question rather than a frozen system-of-record fact).

Prediction. The strong model recovers both the layers and the timing and rarely freezes a live fact.
Weaker models make more LIVE-as-FREEZE errors (the staleness hazard). Whether any tier reliably
separates provenance from pragmatics is the open question the fold surfaces.

Usage (run from study/; Chris runs it):
    python pipeline/design_run_split.py --trials 4
    python pipeline/design_run_split.py --trials 4 --models gpt-5.6-sol --out results/e19_sol.csv
    python pipeline/design_run_split.py --validate        # offline construction + scoring, no API

Writes results/design_run_split.csv (one row per model/trial) and, with --per-fact, a per-fact
breakdown to results/design_run_split_facts.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))
from run import load_dotenv                       # noqa: E402

CASE = "config_cross_domain"
CONF_TAU = 0.80
LAYERS = ["schematic", "instance", "provenance", "pragmatic"]
LAYER_TIMING = {"schematic": "freeze", "provenance": "freeze",
                "instance": "live", "pragmatic": "live"}

# The facts a full cross-domain reconciliation needs, each with its gold LAYER (timing follows the
# layer via LAYER_TIMING). Phrasing keeps the NATURE under test, not a keyword.
FACTS = [
    {"id": "corr-underlay", "layer": "schematic",
     "fact": "That the Meridian 'circuit' and the Cascade 'underlay' denote the same transport object at the seam."},
    {"id": "corr-demarc-identity", "layer": "schematic",
     "fact": "That the Meridian 'hand-off' and the Cascade 'attachment' are one and the same demarcation point."},
    {"id": "cognate-grade", "layer": "schematic",
     "fact": "That the transport 'grade' (an optical protection class) and the IP 'grade' (a class of service) are unrelated and must not be corresponded."},
    {"id": "schema-types", "layer": "schematic",
     "fact": "The kinds and relations of each side's concepts -- what type of thing each is and what it connects to."},
    {"id": "src-rate", "layer": "provenance",
     "fact": "Which system is the authoritative source of record for the committed-rate value (the IP service side sets and polices it; transport only carries it)."},
    {"id": "src-latency", "layer": "provenance",
     "fact": "Which system realises and measures the one-way latency, i.e. whom to ask for the authoritative figure (the transport side)."},
    {"id": "val-latency-now", "layer": "instance",
     "fact": "The one-way latency figure the circuit is actually delivering right now."},
    {"id": "state-protection-now", "layer": "instance",
     "fact": "Whether the circuit is currently running on its primary path or has switched to its protection path."},
    {"id": "deliverability-now", "layer": "pragmatic",
     "fact": "Whether the transport side currently has capacity available to deliver a newly requested realisation."},
    {"id": "use-intent", "layer": "pragmatic",
     "fact": "The broader business use-intent behind the service -- what it is for -- which lives in the application and operators, not the interface."},
    {"id": "demarc-change-resolution", "layer": "pragmatic",
     "fact": "The specific value to adopt for a proposed change to the co-owned demarcation, which requires a fresh bilateral negotiation."},
]

SYSTEM = (
    "You are planning how to obtain the facts a cross-domain reconciliation needs, between a Meridian "
    "transport (optical) system (X) and a Cascade IP/VPN system (Y) that meet at one seam. A lifted "
    "semantic model has four layers:\n"
    " - SCHEMATIC: the fixed ontology -- which concept is which, their kinds and relations, and the "
    "distinctions that keep them apart.\n"
    " - INSTANCE: a concrete current value or record that populates a concept (a measured figure, a "
    "present operational state).\n"
    " - PROVENANCE: who is the authoritative SOURCE OF RECORD for a fact -- who asserts, realises or "
    "measures it, i.e. WHOM TO ASK -- independent of what the current value is.\n"
    " - PRAGMATIC: what a thing is FOR, whether it can be delivered now, and points that need a fresh "
    "negotiation -- the live use-context.\n"
    "For EACH fact, give its LAYER (one of: schematic, instance, provenance, pragmatic), and its "
    "TIMING: \"freeze\" if it is fixed and can be extracted ONCE ahead of use and relied on, or "
    "\"live\" if it must be resolved or refreshed at RUN TIME because it drifts, reflects current "
    "state, or needs a fresh judgement. Add a confidence in [0,1] and a one-line reason. Return one "
    "entry per fact id."
)


def _schemas():
    from pydantic import BaseModel

    class FactVerdict(BaseModel):
        id: str
        layer: str
        timing: str
        confidence: float
        reason: str

    class SplitOut(BaseModel):
        verdicts: list[FactVerdict]

    return SplitOut


def _norm_timing(c: str) -> str:
    c = (c or "").strip().lower()
    if "freeze" in c or "design" in c or "static" in c:
        return "freeze"
    if "live" in c or "run" in c or "dynamic" in c:
        return "live"
    return c


def _norm_layer(c: str) -> str:
    c = (c or "").strip().lower()
    for L in LAYERS:
        if L in c:
            return L
    if "tbox" in c or "schema" in c or "ontolog" in c:
        return "schematic"
    if "abox" in c or "record" in c or "value" in c:
        return "instance"
    if "source" in c or "record" in c or "assert" in c:
        return "provenance"
    if "use" in c or "intent" in c or "context" in c:
        return "pragmatic"
    return c


COLUMNS = ["case", "model", "trial", "n_facts", "layer_accuracy", "timing_accuracy",
           "freeze_as_live", "live_as_freeze", "live_as_freeze_confident",
           "provenance_as_pragmatic", "gold_freeze_fraction", "mean_conf", "reasoning_tokens"]
FACT_COLUMNS = ["case", "model", "trial", "fact_id", "gold_layer", "gold_timing",
                "layer", "timing", "confidence", "layer_correct", "timing_correct"]


def score(verdicts, per_fact_sink=None, model="", trial=0):
    gold_layer = {f["id"]: f["layer"] for f in FACTS}
    gold_timing = {f["id"]: LAYER_TIMING[f["layer"]] for f in FACTS}
    n = len(FACTS)
    layer_correct = timing_correct = 0
    freeze_as_live = live_as_freeze = live_as_freeze_conf = prov_as_prag = 0
    confs = []
    for f in FACTS:
        fid = f["id"]
        v = verdicts.get(fid)
        lay = _norm_layer(v["layer"]) if v else "?"
        tim = _norm_timing(v["timing"]) if v else "?"
        conf = float(v["confidence"]) if v and v.get("confidence") is not None else 0.0
        confs.append(conf)
        gl, gt = gold_layer[fid], gold_timing[fid]
        lok, tok = int(lay == gl), int(tim == gt)
        layer_correct += lok
        timing_correct += tok
        if gt == "freeze" and tim == "live":
            freeze_as_live += 1
        if gt == "live" and tim == "freeze":
            live_as_freeze += 1
            if conf >= CONF_TAU:
                live_as_freeze_conf += 1
        if gl == "provenance" and lay == "pragmatic":
            prov_as_prag += 1
        if per_fact_sink is not None:
            per_fact_sink.append({"case": CASE, "model": model, "trial": trial, "fact_id": fid,
                                  "gold_layer": gl, "gold_timing": gt, "layer": lay, "timing": tim,
                                  "confidence": conf, "layer_correct": lok, "timing_correct": tok})
    gold_freeze = sum(1 for t in gold_timing.values() if t == "freeze")
    return {
        "case": CASE, "model": str(model), "trial": trial, "n_facts": n,
        "layer_accuracy": round(layer_correct / n, 3),
        "timing_accuracy": round(timing_correct / n, 3),
        "freeze_as_live": freeze_as_live, "live_as_freeze": live_as_freeze,
        "live_as_freeze_confident": live_as_freeze_conf,
        "provenance_as_pragmatic": prov_as_prag,
        "gold_freeze_fraction": round(gold_freeze / n, 3),
        "mean_conf": round(mean(confs), 3) if confs else "",
    }


def _run_key(r):
    return (str(r["model"]), str(r["trial"]))


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    if rows:
        counts = {L: sum(1 for f in FACTS if f["layer"] == L) for L in LAYERS}
        layer_str = "  ".join(f"{L}={counts[L]}" for L in LAYERS)
        print(f"\ngold freeze fraction: {rows[0]['gold_freeze_fraction']}  (layers: {layer_str})"
              f"  |  {counts['provenance']} provenance facts")
    print("\n  model          layer-acc  timing-acc   freeze->live  live->freeze(conf)  prov->prag")
    for model in models:
        sel = [r for r in rows if r["model"] == model]
        if not sel:
            continue
        def mn(k):
            xs = [float(x[k]) for x in sel if x[k] not in ("", None)]
            return sum(xs) / len(xs) if xs else 0.0
        print(f"  {model:<14} {mn('layer_accuracy'):.2f}       {mn('timing_accuracy'):.2f}"
              f"        {mn('freeze_as_live'):.2f}          {mn('live_as_freeze'):.2f} "
              f"({mn('live_as_freeze_confident'):.2f})          {mn('provenance_as_pragmatic'):.2f}")
    print()


def validate() -> int:
    ok = True
    layers = [f["layer"] for f in FACTS]
    timings = [LAYER_TIMING[f["layer"]] for f in FACTS]
    print(f"E19 construction check ({CASE}):")
    print(f"  facts: {len(FACTS)}  " + "  ".join(f"{L}={layers.count(L)}" for L in LAYERS))
    print(f"  timing: freeze={timings.count('freeze')} live={timings.count('live')}")
    ids = [f["id"] for f in FACTS]
    ids_ok = len(ids) == len(set(ids))
    layers_ok = all(l in LAYERS for l in layers)
    has_prov = layers.count("provenance") >= 1
    print(f"  ids unique: {ids_ok}   layers valid: {layers_ok}   provenance present: {has_prov}")
    ok = ok and ids_ok and layers_ok and has_prov
    # perfect
    perfect = {f["id"]: {"layer": f["layer"], "timing": LAYER_TIMING[f["layer"]], "confidence": 0.9}
               for f in FACTS}
    r = score(perfect)
    print(f"  perfect: layer-acc={r['layer_accuracy']} timing-acc={r['timing_accuracy']} "
          f"l->f={r['live_as_freeze']} prov->prag={r['provenance_as_pragmatic']}")
    ok = ok and r["layer_accuracy"] == 1.0 and r["timing_accuracy"] == 1.0 and r["live_as_freeze"] == 0
    # staleness error + provenance-as-pragmatic conflation
    bad = dict(perfect)
    liveid = next(f["id"] for f in FACTS if LAYER_TIMING[f["layer"]] == "live")
    bad[liveid] = {"layer": "schematic", "timing": "freeze", "confidence": 0.95}
    provid = next(f["id"] for f in FACTS if f["layer"] == "provenance")
    bad[provid] = {"layer": "pragmatic", "timing": "freeze", "confidence": 0.8}
    r2 = score(bad)
    print(f"  seeded errors: l->f={r2['live_as_freeze']} (conf {r2['live_as_freeze_confident']}) "
          f"prov->prag={r2['provenance_as_pragmatic']}")
    ok = ok and r2["live_as_freeze"] == 1 and r2["live_as_freeze_confident"] == 1 \
        and r2["provenance_as_pragmatic"] == 1
    nrm = (_norm_timing("Design-time") == "freeze" and _norm_timing("RUN TIME") == "live"
           and _norm_layer("Provenance (source of record)") == "provenance"
           and _norm_layer("TBox schematic") == "schematic")
    print(f"  normalisers: {nrm}")
    ok = ok and nrm
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="results/design_run_split.csv")
    ap.add_argument("--per-fact", action="store_true",
                    help="also write a per-fact breakdown to results/design_run_split_facts.csv")
    ap.add_argument("--validate", action="store_true",
                    help="offline: check the fact set and scoring, no API calls")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.validate:
        return validate()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import OpenAIAgentStack
    SplitOut = _schemas()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    trials = max(1, args.trials)
    user = json.dumps({"facts": [{"id": f["id"], "fact": f["fact"]} for f in FACTS]}, indent=2)

    out = ROOT / args.out
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)
    facts_out = ROOT / "results" / "design_run_split_facts.csv"

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {_run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    total = len(models) * trials
    print(f"E19 design/run split: {len(models)} model(s) x {trials} trials = {total} runs",
          file=sys.stderr)

    fh = writer = None
    per_fact_rows = []
    if write:
        fresh_file = args.fresh or not out.exists()
        fh = out.open("w" if fresh_file else "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh_file:
            writer.writeheader()
            fh.flush()
            rows, done = [], set()

    i = 0
    for model in models:
        stack = OpenAIAgentStack(use_reference=False, model=model)
        client = stack._get_client()
        for t in range(trials):
            i += 1
            key = (str(model), str(t))
            if key in done:
                print(f"[{i}/{total}] {model} / trial {t+1}  (captured, skip)", file=sys.stderr)
                continue
            print(f"[{i}/{total}] {model} / trial {t+1}", file=sys.stderr, flush=True)
            try:
                completion = client.chat.completions.parse(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    response_format=SplitOut,
                )
                parsed = completion.choices[0].message.parsed
                verdicts = {v.id: {"layer": v.layer, "timing": v.timing, "confidence": v.confidence}
                            for v in parsed.verdicts}
                usage = getattr(completion, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = getattr(details, "reasoning_tokens", None)
            except Exception as e:
                print(f"      ! error: {e}", file=sys.stderr, flush=True)
                continue
            sink = per_fact_rows if args.per_fact else None
            r = score(verdicts, per_fact_sink=sink, model=model, trial=t)
            r["reasoning_tokens"] = reasoning if reasoning else ""
            rows.append(r)
            done.add(key)
            if writer is not None:
                writer.writerow(r)
                fh.flush()
            print(f"      layer {r['layer_accuracy']} timing {r['timing_accuracy']} "
                  f"live->freeze {r['live_as_freeze']}(conf {r['live_as_freeze_confident']}) "
                  f"prov->prag {r['provenance_as_pragmatic']}", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    if args.per_fact and per_fact_rows and write:
        with facts_out.open("w", newline="") as ff:
            w = csv.DictWriter(ff, fieldnames=FACT_COLUMNS)
            w.writeheader()
            w.writerows(per_fact_rows)
        print(f"Wrote {facts_out.relative_to(ROOT)} ({len(per_fact_rows)} per-fact rows)")
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
