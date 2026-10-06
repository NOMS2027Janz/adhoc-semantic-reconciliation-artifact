#!/usr/bin/env python3
"""E16 -- inert vs cognitive source, head to head: does making a thin source cognitive
recover the meaning a static lift cannot reach?

The production question the reviews keep returning to is not whether cognition can reconcile
two well-described models -- the study already shows that -- but whether the *inputs* can be
produced when the source representation is thin: careless YANG descriptions, or a REST payload
whose meaning lives in the application, not the interface. This experiment isolates the one
variable that the walk-through argues *selects* the production strategy: whether the source is
INERT (a static representation that can only offer what it encodes) or COGNITIVE (a live agent
that can be interrogated for what the representation omits).

Construction. We reuse `config_cross_domain` -- two private, bespoke models (Meridian transport
OSS vs Cascade IP/VPN controller) with no shared standard, whose seam carries two *non-obvious*
correspondences (m.circuit<->c.underlay, m.handoff<->c.attachment: different words, same thing)
and one planted cross-domain false cognate (m.grade<->c.grade: same word, unrelated things --
a transport protection class vs an IP class of service). Model B (Cascade) is fully described
throughout. Model A (Meridian) is the thin side, serialised three ways -- an inert/cognitive/rich
ladder in which the ONLY thing that varies is how A's meaning is supplied:

  * inert     -- A carries label, kind, structural relations, and concrete instances, but NO
                 gloss, synonyms, or example. The meaning is not written into the representation;
                 the agent must reconstruct it from structure, data, and B's descriptions. This
                 models the thin real interface Brad names.
  * cognitive -- the same thin A, PLUS an `explanation` per concept volunteered by a live source
                 agent that owns Model A and understands it (it is interrogated once per concept;
                 its answers are cached). The meaning is supplied live, on demand -- never frozen
                 into the static interface. This is the inert->cognitive shift, made operational.
  * rich      -- A fully described (gloss, synonyms, example), the lift-succeeded upper bound:
                 everything the source could ever carry, statically present.

This is crossed with a REFERENCE mode, which separates the two meaning-supply routes the programme
names. Reference OFF: no shared standard, meaning comes only from A's own content (thin, elicited,
or rich). Reference ON: the case's ad-hoc reference is present as a GOVERNED OVERLAY -- Model B is
bound to it, the thin Model A is not, and the agent binds A's concepts to the overlay by matching.
The overlay is Brad's proposed route where the source cannot be changed; crossing it with the
condition asks whether, given the overlay, source cognition still matters, and whether the overlay
supplies exactly the binding a strong reconciler otherwise withholds. Across the capability ladder
(sol/mini/nano). Reported per condition/ref-mode/model: precision, recall, whether it recovered the
two hard seams (seam_recall), whether it took the grade<->grade cognate, confident-error rate,
residual. The source agent is a fixed strong model so that "the meaning is available" is held
constant across the reconciler ladder -- what varies is the reconciler.

Prediction. inert misses the non-obvious seams and takes the grade cognate (a label alone cannot
tell circuit from underlay, or a transport grade from a service grade); cognitive ~ rich, both
recovering the seams and blocking the cognate. If so, the inert->cognitive shift -- not static
lift richness -- is what recovers the gap: you do not extract the meaning at design time, you
make the source cognitive and elicit it at run time.

Usage:
    # 1) prime the source agent's explanations once (single process; writes the shared cache):
    python inert_vs_cognitive.py --prime-source
    # 2) then the reconciler ladder, both reference modes (safe to split per model across terminals):
    python inert_vs_cognitive.py --trials 4 --fresh
    python inert_vs_cognitive.py --trials 4 --fresh --models gpt-5.6-sol --out results/e16_sol.csv
    # one arm only, if wanted:
    python inert_vs_cognitive.py --trials 4 --ref-modes on --models gpt-5.6-sol --out results/e16b_sol.csv
    # offline construction check, no API, no cost:
    python inert_vs_cognitive.py --validate

Writes results/inert_vs_cognitive.csv (one row per run) and, for the cognitive condition,
results/e16_source_gloss.json (the source agent's cached explanations, keyed by source model).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reconcile.model import Case                 # noqa: E402
from reconcile.metrics import score              # noqa: E402
from run import load_dotenv                       # noqa: E402

DEFAULT_CASE = "config_cross_domain"
CONDITIONS = ["inert", "cognitive", "rich"]
DEFAULT_SOURCE_MODEL = "gpt-5.6-sol"

# the two non-obvious seams per case: different labels, same thing. A label pass cannot bind these;
# only meaning (reconstructed, elicited, or authored) can. seam_recall over these is the sharp signal.
CASE_HARD_SEAMS = {
    "config_cross_domain": [frozenset(("m.circuit", "c.underlay")), frozenset(("m.handoff", "c.attachment"))],
    "config_rest": [frozenset(("p.product", "s.service")), frozenset(("p.config", "s.char"))],
}
# module global, set from --case in main()/validate(); default is the flagship case's seams.
HARD_SEAMS = CASE_HARD_SEAMS[DEFAULT_CASE]

SOURCE_CACHE = "results/e16_source_gloss.json"

# ---------------------------------------------------------------------------
# per-side serialisation: A varies by condition; B is always fully described.
# ---------------------------------------------------------------------------
def _a_concept(c, condition: str, explanation: str | None = None) -> dict:
    if condition == "rich":
        return {"id": c.id, "label": c.label, "kind": c.kind,
                "synonyms": list(c.synonyms), "gloss": c.gloss, "example": c.example,
                "relations": [dict(r) for r in c.relations], "instances": list(c.instances)}
    # inert base: structure and data, but no self-explanation written into the representation
    d = {"id": c.id, "label": c.label, "kind": c.kind,
         "relations": [dict(r) for r in c.relations], "instances": list(c.instances)}
    if condition == "cognitive":
        d["explanation"] = explanation or ""
    return d


def _b_concept(c) -> dict:
    return {"id": c.id, "label": c.label, "kind": c.kind,
            "synonyms": list(c.synonyms), "gloss": c.gloss, "example": c.example,
            "relations": [dict(r) for r in c.relations], "instances": list(c.instances)}


NOTE = {
    "inert": (
        "\n\nModel A is a THIN interface: each of its concepts is given with a label, a kind, its "
        "structural relations, and concrete instances, but NO gloss, synonyms, or example -- its "
        "meaning is not written into the representation. Reconstruct what each Model A concept "
        "denotes from its structure, its instances, and Model B's fully described concepts. A "
        "label alone can mislead -- two concepts sharing a word may denote unrelated things -- so "
        "judge by the roles the relations and instances reveal, not by the label. Model B is fully "
        "described."
    ),
    "cognitive": (
        "\n\nModel A's interface is thin -- no gloss, synonyms, or example is written into it -- "
        "but the live system that OWNS Model A has been interrogated and volunteers an "
        "'explanation' for each concept. Treat that explanation as authoritative meaning supplied "
        "by the system itself, on demand. Model B is fully described. Judge correspondences by the "
        "supplied meaning and the structure, not by labels alone: two concepts sharing a word may "
        "denote unrelated things."
    ),
    "rich": (
        "\n\nBoth models are fully described: each concept carries its gloss, synonyms, and "
        "canonical example alongside its structure and instances. Judge correspondences by "
        "meaning, not by labels alone -- two concepts sharing a word may denote unrelated things."
    ),
}

# The reference-on arm: the ad-hoc reference is the GOVERNED OVERLAY (Brad's proposed route where
# the source cannot be changed). Model B (live) carries its binding; the thin Model A does not, so
# the agent must bind A's concepts to the overlay by matching -- which is exactly how a published
# reference supplies the meaning a thin source cannot volunteer. Crossed with the inert/cognitive/
# rich condition, this separates the two meaning-supply routes: the overlay (reference on) versus
# the live source (the cognitive condition). Same note for all conditions; only A's self-description
# varies, so the note stays honest whether A is thin or rich.
REF_NOTE = (
    "\n\nYou are also given a shared published reference -- a governed overlay: identity-only "
    "entries, each with an id, definition, and canonical example. Model B's concepts carry their "
    "reference binding ('ref'); Model A's concepts do NOT. Decide which reference entry each Model "
    "A concept denotes by matching it to the entry whose definition and example fit, then "
    "correspond concepts that map to the SAME entry. The overlay is the shared category that "
    "authorises a cross-domain correspondence and pre-empts false cognates: two concepts whose "
    "labels look alike but fit different entries -- or fit no entry at all -- do not correspond."
)

SOURCE_SYSTEM = (
    "You ARE the system named below, and you own the concept described. You understand exactly "
    "what it is and what it is for. A peer system is reconciling its own model against yours and "
    "has asked you to explain this one concept. In two or three sentences, explain what the "
    "concept denotes and its role -- what it is for, what it connects to, and how to tell it apart "
    "from things that merely share its name. Explain in your own words, from your understanding of "
    "your own domain; do not guess at or name the peer's terminology."
)


def build_user(a, b, condition: str, glosses: dict,
               reference=None, use_reference: bool = False) -> str:
    def bconc(c):
        d = _b_concept(c)
        if use_reference:
            d["ref"] = c.ref            # the live side carries its overlay binding
        return d
    payload = {
        "model_A": {"system": a.system, "dialect": a.dialect, "thin": condition != "rich",
                    "concepts": [_a_concept(c, condition, glosses.get(c.id))
                                 for c in a.concepts]},   # thin side NOT pre-bound to the overlay
        "model_B": {"system": b.system, "dialect": b.dialect, "thin": False,
                    "concepts": [bconc(c) for c in b.concepts]},
    }
    if use_reference and reference is not None:
        payload["reference"] = [
            {"id": e.id, "definition": e.definition, "example": e.example}
            for e in reference.entries
        ]
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# the source agent: one interrogation per Model A concept, cached and shared across runs.
# ---------------------------------------------------------------------------
def _cache_path(out_cache: str) -> Path:
    return ROOT / out_cache


def load_cache(out_cache: str) -> dict:
    p = _cache_path(out_cache)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def source_explain(client, source_model: str, a) -> dict:
    """Interrogate the source agent once per Model A concept; return {concept_id: explanation}."""
    out = {}
    for c in a.concepts:
        truth = {"id": c.id, "label": c.label, "kind": c.kind, "synonyms": list(c.synonyms),
                 "gloss": c.gloss, "example": c.example,
                 "relations": [dict(r) for r in c.relations], "instances": list(c.instances)}
        user = ("Your system: " + a.system + " (" + a.dialect + ").\n"
                "The concept you own:\n" + json.dumps(truth, indent=2))
        completion = client.chat.completions.create(
            model=source_model,
            messages=[{"role": "system", "content": SOURCE_SYSTEM},
                      {"role": "user", "content": user}],
        )
        out[c.id] = (completion.choices[0].message.content or "").strip()
        print(f"    [source:{source_model}] {c.id}: {out[c.id][:80]}...", file=sys.stderr, flush=True)
    return out


def ensure_glosses(client, source_model: str, a, out_cache: str) -> dict:
    """Return the source model's explanations for every A concept, generating and caching any
    that are missing. Atomic write keeps parallel workers from corrupting the shared cache."""
    cache = load_cache(out_cache)
    have = cache.get(source_model, {})
    missing = [c.id for c in a.concepts if not have.get(c.id)]
    if missing:
        print(f"  priming source model {source_model} for {len(missing)} concept(s)...",
              file=sys.stderr, flush=True)
        fresh = source_explain(client, source_model, a)
        # merge (keep any already present; overwrite only the missing/blank)
        merged = dict(have)
        for cid, text in fresh.items():
            if not merged.get(cid):
                merged[cid] = text
        cache[source_model] = merged
        _atomic_write_json(_cache_path(out_cache), cache)
        have = merged
    return {c.id: have.get(c.id, "") for c in a.concepts}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
COLUMNS = ["case", "model", "condition", "uses_reference", "source_model", "trial",
           "precision", "recall", "seam_recall", "took_grade_cognate",
           "surviving_false_cognates", "residual", "confident_errors", "confident_error_rate",
           "conf_wrong_mean", "calibration_gap", "source_queries", "reasoning_tokens", "missed"]


def _row(case, model, condition, use_reference, source_model, trial, rec, gold, source_queries):
    m = score(rec, gold)
    proposed = set(rec.proposed)
    correct = gold.correct_pairs
    missed = correct - proposed
    seam_hit = sum(1 for s in HARD_SEAMS if s in proposed)
    return {
        "case": case, "model": str(model), "condition": condition,
        "uses_reference": str(bool(use_reference)),
        "source_model": source_model if condition == "cognitive" else "",
        "trial": trial,
        "precision": m["precision"], "recall": m["recall"],
        "seam_recall": round(seam_hit / len(HARD_SEAMS), 3),
        "took_grade_cognate": int(len(proposed & gold.false_cognate_pairs) > 0),
        "surviving_false_cognates": m["surviving_false_cognates"], "residual": m["residual"],
        "confident_errors": m["confident_errors"], "confident_error_rate": m["confident_error_rate"],
        "conf_wrong_mean": m["conf_wrong_mean"], "calibration_gap": m["calibration_gap"],
        "source_queries": source_queries,
        "reasoning_tokens": rec.effort.get("reasoning_tokens", ""),
        "missed": ";".join(sorted("~".join(sorted(p)) for p in missed)),
    }


def _run_key(r):
    return (str(r["model"]), r["condition"], str(r.get("uses_reference", "False")), str(r["trial"]))


def summarize(rows):
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    ref_modes = sorted({str(r.get("uses_reference", "False")) for r in rows})
    for model in models:
        for ref in ref_modes:
            mr = [r for r in rows if r["model"] == model
                  and str(r.get("uses_reference", "False")) == ref]
            if not mr:
                continue
            tag = "reference ON (overlay)" if ref == "True" else "reference OFF"
            print(f"\n{'#'*10} {model}  --  {tag} {'#'*10}")
            print("  condition    precision   recall   seam-recall   grade-cognate   confident-err")
            for cond in CONDITIONS:
                sel = [r for r in mr if r["condition"] == cond]
                if not sel:
                    continue
                def mean(k):
                    xs = [float(x[k]) for x in sel if x[k] not in ("", None)]
                    return sum(xs) / len(xs) if xs else 0.0
                print(f"   {cond:<10}  {mean('precision'):.2f}       {mean('recall'):.2f}"
                      f"     {mean('seam_recall'):.2f}         {mean('took_grade_cognate'):.2f}"
                      f"           {mean('confident_errors'):.2f}")
    print()


# ---------------------------------------------------------------------------
# offline validation: build all three condition payloads and check the construction, no API.
# ---------------------------------------------------------------------------
def validate(case_name=DEFAULT_CASE) -> int:
    global HARD_SEAMS
    HARD_SEAMS = CASE_HARD_SEAMS[case_name]
    from reconcile.stacks.agent_openai import ReconcileResult, _Correspondence, OpenAIAgentStack
    case = Case.load(ROOT / "benchmark" / "cases" / case_name)
    a, b, gold = case.model_a, case.model_b, case.gold
    ok = True
    print("E16 construction check (config_cross_domain):")
    print(f"  A (thin side) = {a.system}: {[c.id for c in a.concepts]}")
    print(f"  B (full)      = {b.system}: {[c.id for c in b.concepts]}")
    print(f"  gold: {len(gold.correspondences)} correspondences, "
          f"{len(gold.false_cognates)} false cognate(s)")
    seams_in_gold = all(s in gold.correct_pairs for s in HARD_SEAMS)
    print(f"  hard seams present in gold: {seams_in_gold} "
          f"({sorted('~'.join(sorted(s)) for s in HARD_SEAMS)})")
    ok = ok and seams_in_gold

    stub = {c.id: f"STUB explanation of {c.label}" for c in a.concepts}
    for cond in CONDITIONS:
        user = json.loads(build_user(a, b, cond, stub))
        a0 = user["model_A"]["concepts"][0]           # m.circuit
        has_meaning = "gloss" in a0
        has_expl = "explanation" in a0
        has_struct = "relations" in a0 and "instances" in a0
        b0 = user["model_B"]["concepts"][0]
        b_full = all(k in b0 for k in ("gloss", "synonyms", "example"))
        fields = sorted(a0.keys())
        print(f"  [{cond:<9}] A fields={fields}")
        if cond == "inert":
            good = (not has_meaning) and (not has_expl) and has_struct and b_full
        elif cond == "cognitive":
            good = (not has_meaning) and has_expl and has_struct and a0["explanation"] and b_full
        else:
            good = has_meaning and has_struct and b_full
        print(f"             field invariant holds: {good}; B fully described: {b_full}")
        ok = ok and good

    # reference-on arm: overlay present, B bound, thin A not bound
    for cond in CONDITIONS:
        user = json.loads(build_user(a, b, cond, stub, case.reference, use_reference=True))
        a0 = user["model_A"]["concepts"][0]
        b0 = user["model_B"]["concepts"][0]
        n_entries = len(user.get("reference", []))
        a_unbound = "ref" not in a0
        b_bound = "ref" in b0 and b0["ref"]
        print(f"  [ref-on {cond:<7}] reference entries={n_entries}; A unbound={a_unbound}; "
              f"B bound={bool(b_bound)}")
        good = (n_entries == len(case.reference.entries)) and a_unbound and bool(b_bound)
        ok = ok and good

    # exercise the scoring path with a mock result (all correct + this case's planted cognate)
    stack = OpenAIAgentStack(use_reference=False, model="mock")
    corr = [_Correspondence(a_id=c["a"], b_id=c["b"], confidence=0.9, rationale="mock")
            for c in gold.correspondences]
    trap = gold.false_cognates[0]
    corr.append(_Correspondence(a_id=trap["a"], b_id=trap["b"], confidence=0.85, rationale="mock trap"))
    result = ReconcileResult(correspondences=corr, residual_a=[], residual_b=[])
    rec = stack.to_reconciliation(result, a, b, {"reasoning_tokens": 0}, "both_cognitive")
    r = _row(case_name, "mock", "inert", False, DEFAULT_SOURCE_MODEL, 0, rec, gold, 0)
    print(f"  scoring self-check (all-correct + planted cognate): precision={r['precision']} "
          f"recall={r['recall']} seam_recall={r['seam_recall']} "
          f"took_grade={r['took_grade_cognate']} surv_fc={r['surviving_false_cognates']}")
    ok = ok and (r["recall"] == 1.0 and r["seam_recall"] == 1.0 and r["took_grade_cognate"] == 1
                 and r["precision"] < 1.0)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-5.6-sol,gpt-5-mini,gpt-5-nano",
                    help="comma-separated reconciler model list (the capability ladder)")
    ap.add_argument("--case", default=DEFAULT_CASE, choices=sorted(CASE_HARD_SEAMS),
                    help="benchmark case to run (must define its two non-obvious seams in CASE_HARD_SEAMS)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma-separated subset of inert,cognitive,rich")
    ap.add_argument("--ref-modes", default="off,on",
                    help="'off,on' runs both the no-overlay arm and the governed-overlay (ad-hoc "
                         "reference) arm; 'off' or 'on' runs just one")
    ap.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL,
                    help="model that plays the live source agent for the cognitive condition "
                         "(fixed strong by default, so 'meaning is available' is held constant)")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="results/inert_vs_cognitive.csv",
                    help="output CSV (relative to study/); give each parallel terminal its own "
                         "file, e.g. results/e16_sol.csv, then merge")
    ap.add_argument("--source-cache", default=SOURCE_CACHE,
                    help="shared JSON cache of the source agent's explanations")
    ap.add_argument("--prime-source", action="store_true",
                    help="only generate+cache the source agent's explanations, then exit "
                         "(run once before parallel reconciler runs)")
    ap.add_argument("--validate", action="store_true",
                    help="offline: check the three-condition construction and scoring, no API calls")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results file and start over (default: resume)")
    args = ap.parse_args()

    if args.validate:
        return validate(args.case)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from reconcile.stacks.agent_openai import SYSTEM, ReconcileResult, OpenAIAgentStack

    global HARD_SEAMS
    case_name = args.case
    HARD_SEAMS = CASE_HARD_SEAMS[case_name]
    case = Case.load(ROOT / "benchmark" / "cases" / case_name)
    a, b, gold = case.model_a, case.model_b, case.gold
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            print(f"unknown condition {c!r}; choose from {CONDITIONS}", file=sys.stderr)
            return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ref_modes = [tok.strip() == "on" for tok in args.ref_modes.split(",") if tok.strip()]
    trials = max(1, args.trials)
    reference = case.reference

    # per-case source-gloss cache: the default cache is keyed to the case so a second case cannot
    # reuse the first case's cached explanations (different concepts). An explicit --source-cache wins.
    source_cache = args.source_cache
    if source_cache == SOURCE_CACHE and case_name != DEFAULT_CASE:
        source_cache = f"results/e16_source_gloss_{case_name}.json"

    # source agent: prime the cognitive condition's explanations (shared, cached)
    glosses = {}
    if "cognitive" in conditions or args.prime_source:
        prime_stack = OpenAIAgentStack(use_reference=False, model=args.source_model)
        client = prime_stack._get_client()
        glosses = ensure_glosses(client, args.source_model, a, source_cache)
        if args.prime_source:
            print(f"Primed source model {args.source_model}: "
                  f"{sum(1 for v in glosses.values() if v)}/{len(glosses)} concepts cached "
                  f"in {source_cache}")
            return 0

    out = ROOT / args.out
    write = not args.no_write
    if write:
        out.parent.mkdir(exist_ok=True)

    rows, done = [], set()
    if write and out.exists() and not args.fresh:
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        done = {_run_key(r) for r in rows}
        print(f"Resuming: {len(done)} runs already captured in {out.name}", file=sys.stderr)

    total = len(models) * len(conditions) * len(ref_modes) * trials
    print(f"E16 inert-vs-cognitive: {len(models)} model(s) x {len(conditions)} conditions x "
          f"{len(ref_modes)} ref-mode(s) x {trials} trials = {total} runs", file=sys.stderr)

    fh = writer = None
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
        for use_ref in ref_modes:
            stack = OpenAIAgentStack(use_reference=use_ref, model=model)
            client = stack._get_client()
            for condition in conditions:
                user = build_user(a, b, condition, glosses, reference, use_ref)
                system = SYSTEM + NOTE[condition] + (REF_NOTE if use_ref else "")
                source_queries = len(a.concepts) if condition == "cognitive" else 0
                reftag = "ref-on" if use_ref else "ref-off"
                for t in range(trials):
                    i += 1
                    key = (str(model), condition, str(bool(use_ref)), str(t))
                    if key in done:
                        print(f"[{i}/{total}] {model} / {condition} / {reftag} / trial {t+1}  "
                              f"(captured, skip)", file=sys.stderr, flush=True)
                        continue
                    print(f"[{i}/{total}] {model} / {condition} / {reftag} / trial {t+1}",
                          file=sys.stderr, flush=True)
                    try:
                        t0 = time.time()
                        completion = client.chat.completions.parse(
                            model=model,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                            response_format=ReconcileResult,
                        )
                        elapsed = time.time() - t0
                        result = completion.choices[0].message.parsed
                        usage = getattr(completion, "usage", None)
                        details = getattr(usage, "completion_tokens_details", None)
                        effort = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                            "latency_s": round(elapsed, 2),
                            "model": model,
                        }
                        placement = "one_inert" if use_ref else "both_cognitive"
                        rec = stack.to_reconciliation(result, a, b, effort, placement)
                    except Exception as e:
                        print(f"      ! error: {e}", file=sys.stderr, flush=True)
                        continue
                    r = _row(case_name, model, condition, use_ref, args.source_model, t, rec, gold,
                             source_queries)
                    rows.append(r)
                    done.add(key)
                    if writer is not None:
                        writer.writerow(r)
                        fh.flush()
                    print(f"      prec {r['precision']} rec {r['recall']} seam {r['seam_recall']} "
                          f"grade-cognate {r['took_grade_cognate']} conf.err {r['confident_errors']} "
                          f"missed[{r['missed']}]", file=sys.stderr, flush=True)

    if fh is not None:
        fh.close()
    summarize(rows)
    if write:
        print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
