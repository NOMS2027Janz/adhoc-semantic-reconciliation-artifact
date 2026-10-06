#!/usr/bin/env python3
"""Benchmark packaging tool: validate cases and build the manifest.

The benchmark is the set of cases under ``benchmark/cases/``. Each case is a small
folder of plain-JSON files (see ``benchmark/schema.md``). This tool is the gate and
the index for that set, and it runs entirely offline (no model access):

  python benchmark/pack.py validate          # check every case is well-formed
  python benchmark/pack.py validate <case>   # check one case
  python benchmark/pack.py manifest          # (re)write benchmark/manifest.json

A contributor adds a case folder, runs ``validate`` until it passes, and runs
``manifest`` to register it. ``validate`` exits non-zero on any error, so it can gate
CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASES = Path(__file__).resolve().parent / "cases"

# Which operational setting each case belongs to (for the manifest index).
SETTING = {
    "config_tapi_teas": "1 · configuration", "config_big_hard": "1 · configuration",
    "instance_hard": "1 · configuration (instances)", "verify_hard": "1 · configuration (verification)",
    "scaling_otn": "1 · configuration (scaling)",
    "intent_hard": "2 · intent", "intent_endpoints": "2 · intent (endpoints)",
    "config_cross_domain": "3 · cross-domain",
    "config_observability": "4 · observability", "obs_instance": "4 · observability (instances)",
}


def _load(path: Path):
    """Load one JSON file, returning (data, error-string-or-None)."""
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, f"missing file {path.name}"
    except json.JSONDecodeError as e:
        return None, f"malformed JSON in {path.name}: {e}"


def classify(case_dir: Path) -> str:
    """Name the case family from its signature files."""
    names = {p.name for p in case_dir.glob("*.json")}
    has = lambda n: n in names
    model_a = any(n.startswith("model_a") for n in names)
    model_b = any(n.startswith("model_b") for n in names)
    model_c = any(n.startswith("model_c") for n in names)
    if has("proposals.json") and has("verify_gold.json"):
        return "verify"
    if any(n.startswith("individuals_a") for n in names) and has("instance_gold.json"):
        return "instance"
    if has("catalogue.json") and has("intent_gold.json"):
        return "intent"
    if model_a and model_c and not has("gold.json"):
        return "scaling"
    if model_a and model_b and has("gold.json"):
        return "schema"
    return "unknown"


# Required files per family (globs allowed via startswith on model_*).
REQUIRED = {
    "schema":   ["model_a*", "model_b*", "reference.json", "gold.json"],
    "instance": ["individuals_a*", "individuals_b*", "instance_reference.json", "instance_gold.json"],
    "intent":   ["catalogue.json", "intents.json", "intent_reference.json", "intent_gold.json"],
    "scaling":  ["model_a*", "model_b*", "reference.json"],
    "verify":   ["proposals.json", "verify_gold.json"],
}


def _present(names: set[str], pattern: str) -> bool:
    return pattern[:-1] and any(n.startswith(pattern[:-1]) for n in names) if pattern.endswith("*") \
        else pattern in names


def validate_case(case_dir: Path) -> list[str]:
    """Return a list of error strings for one case ([] means valid)."""
    errs: list[str] = []
    names = {p.name for p in case_dir.glob("*.json")}
    if not names:
        return [f"{case_dir.name}: no JSON files"]

    fam = classify(case_dir)
    if fam == "unknown":
        return [f"{case_dir.name}: unrecognised case family (no known signature files)"]

    # 1. required files present
    for pat in REQUIRED[fam]:
        if not _present(names, pat):
            errs.append(f"{case_dir.name}: required file '{pat}' missing")

    # 2. every JSON parses
    data = {}
    for p in sorted(case_dir.glob("*.json")):
        d, err = _load(p)
        if err:
            errs.append(f"{case_dir.name}: {err}")
        else:
            data[p.name] = d

    # 3. deeper structural checks for the documented schema family
    if fam == "schema" and not errs:
        errs += _check_schema_case(case_dir.name, data, names)
    return errs


def _model_files(names: set[str], side: str) -> list[str]:
    return sorted(n for n in names if n.startswith(f"model_{side}"))


def _check_schema_case(case: str, data: dict, names: set[str]) -> list[str]:
    errs: list[str] = []
    a_name = _model_files(names, "a")[0]
    b_name = _model_files(names, "b")[0]
    a, b = data[a_name], data[b_name]
    ref = data.get("reference.json", {})
    gold = data.get("gold.json", {})

    a_ids = {c["id"] for c in a.get("concepts", [])}
    b_ids = {c["id"] for c in b.get("concepts", [])}
    ref_ids = {e["id"] for e in ref.get("entries", ref if isinstance(ref, list) else [])}

    # unique concept ids
    for side, model, name in [("A", a, a_name), ("B", b, b_name)]:
        ids = [c["id"] for c in model.get("concepts", [])]
        if len(ids) != len(set(ids)):
            errs.append(f"{case}: duplicate concept ids in {name}")

    # relations resolve within the same model
    for side, model, ids, name in [("A", a, a_ids, a_name), ("B", b, b_ids, b_name)]:
        for c in model.get("concepts", []):
            for r in c.get("relations", []):
                if r.get("target") not in ids:
                    errs.append(f"{case}: {name} concept '{c['id']}' relation target "
                                f"'{r.get('target')}' is not a concept in the same model")

    # declared ref bindings resolve to the reference
    if ref_ids:
        for side, model, name in [("A", a, a_name), ("B", b, b_name)]:
            for c in model.get("concepts", []):
                rid = c.get("ref")
                if rid is not None and rid not in ref_ids:
                    errs.append(f"{case}: {name} concept '{c['id']}' ref '{rid}' "
                                f"not found in reference.json")

    # gold correspondences reference real concept ids on each side
    for corr in gold.get("correspondences", []):
        if corr.get("a") not in a_ids:
            errs.append(f"{case}: gold correspondence a='{corr.get('a')}' not a concept in {a_name}")
        if corr.get("b") not in b_ids:
            errs.append(f"{case}: gold correspondence b='{corr.get('b')}' not a concept in {b_name}")

    # false-cognate traps reference real ids too
    for fc in gold.get("false_cognates", []):
        if fc.get("a") not in a_ids or fc.get("b") not in b_ids:
            errs.append(f"{case}: gold false_cognate {fc.get('a')}~{fc.get('b')} "
                        f"references an unknown concept id")
    return errs


def _counts(case_dir: Path, fam: str, data: dict, names: set[str]) -> dict:
    c = {"family": fam, "setting": SETTING.get(case_dir.name, "unassigned"),
         "files": sorted(names)}
    if fam == "schema":
        a = data[_model_files(names, "a")[0]]; b = data[_model_files(names, "b")[0]]
        gold = data.get("gold.json", {})
        c.update(concepts_a=len(a.get("concepts", [])), concepts_b=len(b.get("concepts", [])),
                 correspondences=len(gold.get("correspondences", [])),
                 false_cognates=len(gold.get("false_cognates", [])),
                 has_reference="reference.json" in names,
                 has_pragmatics="pragmatics.json" in names)
    elif fam == "instance":
        gold = data.get("instance_gold.json", {})
        c.update(correspondences=len(gold.get("correspondences", [])),
                 experiment_only=len(gold.get("experiment_only", [])))
    elif fam == "scaling":
        c.update(models=len([n for n in names if n.startswith("model_")]))
    return c


def cmd_validate(which: str | None) -> int:
    dirs = [CASES / which] if which else sorted(p for p in CASES.iterdir() if p.is_dir())
    total_err = 0
    for d in dirs:
        if not d.is_dir():
            print(f"  ! no such case: {d.name}", file=sys.stderr); total_err += 1; continue
        errs = validate_case(d)
        if errs:
            total_err += len(errs)
            print(f"  ✗ {d.name} ({classify(d)}) — {len(errs)} error(s):")
            for e in errs:
                print(f"      {e}")
        else:
            print(f"  ✓ {d.name} ({classify(d)})")
    print(f"\n{'FAIL' if total_err else 'OK'}: {len(dirs)} case(s), {total_err} error(s).")
    return 1 if total_err else 0


def cmd_manifest() -> int:
    entries = []
    for d in sorted(p for p in CASES.iterdir() if p.is_dir()):
        names = {p.name for p in d.glob("*.json")}
        fam = classify(d)
        data = {}
        for p in d.glob("*.json"):
            dd, err = _load(p)
            if not err:
                data[p.name] = dd
        entry = {"case": d.name}
        entry.update(_counts(d, fam, data, names))
        entries.append(entry)
    manifest = {"benchmark": "ad-hoc-semantic-reconciliation",
                "n_cases": len(entries), "cases": entries}
    out = Path(__file__).resolve().parent / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out.relative_to(out.parents[1])} — {len(entries)} case(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate all cases, or one named case")
    v.add_argument("case", nargs="?", default=None)
    sub.add_parser("manifest", help="write benchmark/manifest.json")
    args = ap.parse_args()
    if args.cmd == "validate":
        return cmd_validate(args.case)
    if args.cmd == "manifest":
        return cmd_manifest()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
