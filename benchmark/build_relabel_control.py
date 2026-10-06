#!/usr/bin/env python3
"""Build a memorisation control for the flagship setting-1 case (Brad's point 7).

The flagship case is real, well-known TAPI vs TEAS, so a strong model may be recalling a
TAPI/TEAS mapping it has seen rather than reasoning from the prompt. This script clones
config_tapi_teas into config_relabel_control, structurally IDENTICAL, but with every
recognisable public-standard hook removed:

  * concept ids are neutralised (t.cs -> a05, i.tunnel -> b05, ...), consistently across
    relations, the reference bindings, and the gold, so nothing is scored differently;
  * the reference entry ids are neutralised (connection-service -> r06, ...);
  * every surface string (system, dialect, label, synonyms, gloss, example, reference
    label/definition/example) has its TAPI/TEAS-specific vocabulary swapped for invented,
    semantically-parallel terms, applied consistently so the cross-model surface collisions
    that make the false-cognate traps are preserved.

Generic networking words (node, link, service, topology, layer, endpoint, ...) stay: they are
the reasoning substrate, not a memorised mapping. The gold pairing is preserved up to the id
renaming, so the relabelled case scores by the same rule. Run the agent on both cases and
compare: similar scores => not memorisation; a drop on the relabelled case => recall was helping.

  python benchmark/build_relabel_control.py           # writes benchmark/cases/config_relabel_control/
  python benchmark/pack.py validate                   # then confirm it is well-formed

The script verifies (and refuses to write on failure): (1) the cross-model surface-collision
structure is identical before and after; (2) no banned standard token survives in any surface.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "cases" / "config_tapi_teas"
DST = HERE / "cases" / "config_relabel_control"

# Two-stage, COLLISION-SAFE relabelling. First a couple of standards-name phrases, then a
# strict TOKEN map applied at word boundaries. A uniform token rename preserves every
# cross-model surface collision by construction (a shared token X becomes a shared token X'
# everywhere), which is why we substitute tokens, not phrases. Generic function words are left
# alone so the glosses stay readable; only domain/standard nouns and abbreviations are renamed.
PHRASES = {
    "onf tapi": "Kelvin-TM", "ietf teas": "Rowan-TM",
    "traffic-engineering": "flux-managed", "traffic engineering": "flux-managed",
    "cross-connect": "cross-switch", "cross connect": "cross switch",
}
TOKENS = {
    # standard identity
    "tapi": "kelvin", "teas": "rowan", "onf": "kelvin", "ietf": "rowan", "otn": "zeta",
    "te": "flux",
    # domain nouns (renamed consistently, so collisions are preserved)
    "node": "vertex", "link": "span", "termination": "cap", "service": "offer",
    "tunnel": "conduit", "topology": "fabric", "connection": "circuit",
    "connectivity": "carriage", "adaptation": "bridging", "transitional": "shifting",
    "layer": "tier", "protocol": "stream", "matrix": "grid", "interface": "gateway", "otn": "zeta",
    "supporting": "backing", "domain": "realm", "trail": "run", "rule": "gate", "group": "set",
    # abbreviations
    "tp": "cp", "ttp": "ccp", "sip": "acp", "cep": "cxp", "nep": "hep", "conn": "ckt",
    "tpn": "xpn", "ts": "xs", "tributary": "carrier",
}
# terse fields (instances, relation predicates) carry glued standard fragments the word-boundary
# token map cannot reach (telink, tcep, ODU2, OTU4, ...). These are scrubbed by substring, applied
# AFTER the token map. Readability matters less here, so substring replacement is safe.
FRAGMENTS = [
    ("cross-connect", "cross-switch"), ("otnlabel", "zetatag"), ("telink", "fluxspan"),
    ("tenode", "fluxvertex"), ("tetopo", "fluxfabric"), ("cmatrix", "cgrid"), ("tcep", "ccx"),
    ("tunnel", "conduit"), ("connectivity", "carriage"), ("adaptation", "bridging"),
    ("otu", "crr"), ("odu", "grd"), ("otn", "zeta"), ("ttp", "ccp"), ("nep", "hep"),
    ("sip", "acp"), ("cs-", "co-"), ("te-", "flux-"), ("tp-", "cp-"),
]
# original tokens/fragments that must not survive anywhere in the relabelled record
BANNED = [r"\btapi\b", r"\bteas\b", r"\bietf\b", r"\bonf\b", "otn", "odu", "otu", "tunnel",
          "connectivity", "tributary", "telink", "tenode", "tetopo", "tcep", "cmatrix",
          "otnlabel", "cross-connect", r"\bsip\b", r"\bcep\b", r"\bttp\b", r"\bnep\b", r"\bte-"]

_PHRASE = re.compile("|".join(re.escape(k) for k in sorted(PHRASES, key=len, reverse=True)),
                     re.IGNORECASE)
_TOKEN = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(TOKENS, key=len, reverse=True))
                    + r")\b", re.IGNORECASE)


# OTN signal names (ODU2, OTU4, ODUk, ...) at word boundaries, so glosses stay intact but the
# recognisable signal identities go. Kept separate from the substring scrub so words like
# "module"/"nodule" are never touched.
_SIGNAL = re.compile(r"\b(od|ot)u(\d*)\b", re.IGNORECASE)
_SIGREP = {"od": "grd", "ot": "crr"}


def relabel_text(s: str) -> str:
    """For readable surface: labels, synonyms, glosses, examples, reference text."""
    s = _PHRASE.sub(lambda m: PHRASES[m.group(0).lower()], s)
    s = _SIGNAL.sub(lambda m: _SIGREP[m.group(1).lower()] + m.group(2), s)
    return _TOKEN.sub(lambda m: TOKENS[m.group(0).lower()], s)


def relabel_terse(s: str) -> str:
    """For instances and relation predicates: token map, then substring fragment scrub."""
    s = relabel_text(s)
    low = s
    for frag, rep in FRAGMENTS:
        low = re.sub(re.escape(frag), rep, low, flags=re.IGNORECASE)
    return low


def surface_tokens(concept: dict) -> set:
    text = " ".join([concept.get("label", ""), *concept.get("synonyms", [])]).lower()
    return {t for t in re.split(r"[\s/_-]+", text) if t}


def collision_pairs(a_concepts, b_concepts):
    """(a_id, b_id) pairs whose label+synonym token sets overlap -- the surface structure the
    traps ride on."""
    out = set()
    for ca in a_concepts:
        ta = surface_tokens(ca)
        for cb in b_concepts:
            if ta & surface_tokens(cb):
                out.add((ca["id"], cb["id"]))
    return out


def main() -> int:
    a = json.loads((SRC / "model_a_tapi.json").read_text())
    b = json.loads((SRC / "model_b_teas.json").read_text())
    ref = json.loads((SRC / "reference.json").read_text())
    gold = json.loads((SRC / "gold.json").read_text())

    # id maps, in file order, neutral and standard-free
    id_map = {}
    for i, c in enumerate(a["concepts"], 1):
        id_map[c["id"]] = f"a{i:02d}"
    for i, c in enumerate(b["concepts"], 1):
        id_map[c["id"]] = f"b{i:02d}"
    ref_entries = ref["entries"] if isinstance(ref, dict) else ref
    ref_map = {e["id"]: f"r{i:02d}" for i, e in enumerate(ref_entries, 1)}

    # collisions BEFORE, keyed by the id map so we can compare to AFTER
    before = {(id_map[x], id_map[y]) for x, y in collision_pairs(a["concepts"], b["concepts"])}

    def remap_concept(c: dict) -> dict:
        d = dict(c)
        d["id"] = id_map[c["id"]]
        d["label"] = relabel_text(c.get("label", ""))
        d["synonyms"] = [relabel_text(s) for s in c.get("synonyms", [])]
        if "gloss" in c:
            d["gloss"] = relabel_text(c["gloss"])
        if "example" in c:
            d["example"] = relabel_text(c["example"])
        if c.get("ref") is not None:
            d["ref"] = ref_map.get(c["ref"], c["ref"])
        if "relations" in c:
            d["relations"] = [{**r, "rel": relabel_terse(r.get("rel", "")),
                               "target": id_map.get(r.get("target"), r.get("target"))}
                              for r in c["relations"]]
        if "instances" in c:
            d["instances"] = [relabel_terse(x) for x in c["instances"]]
        # kind: structural label (node/service/...) -> unchanged
        return d

    def remap_model(m: dict) -> dict:
        d = dict(m)
        for k, v in m.items():  # relabel every top-level field (system, dialect, modules, ...)
            if k == "concepts":
                continue
            if isinstance(v, str):
                d[k] = relabel_terse(v)
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                d[k] = [relabel_terse(x) for x in v]
        d["concepts"] = [remap_concept(c) for c in m["concepts"]]
        return d

    a2, b2 = remap_model(a), remap_model(b)

    ref2_entries = []
    for e in ref_entries:
        e2 = dict(e)
        e2["id"] = ref_map[e["id"]]
        for k in ("label", "definition", "example"):
            if k in e:
                e2[k] = relabel_text(e[k])
        if "synonyms" in e:
            e2["synonyms"] = [relabel_text(s) for s in e["synonyms"]]
        ref2_entries.append(e2)
    ref2 = {**ref, "entries": ref2_entries} if isinstance(ref, dict) else ref2_entries

    def remap_ids_in_gold(g):
        g2 = dict(g)
        g2["correspondences"] = [{"a": id_map[c["a"]], "b": id_map[c["b"]],
                                  "ref": ref_map.get(c.get("ref"), c.get("ref"))}
                                 for c in g.get("correspondences", [])]
        if "false_cognates" in g:
            g2["false_cognates"] = [{**c, "a": id_map.get(c["a"], c["a"]),
                                     "b": id_map.get(c["b"], c["b"])} for c in g["false_cognates"]]
        if "residual" in g:
            res = g["residual"]
            g2["residual"] = {
                "a_only": [{**x, "id": id_map.get(x["id"], x["id"])} if isinstance(x, dict)
                           else id_map.get(x, x) for x in res.get("a_only", [])],
                "b_only": [{**x, "id": id_map.get(x["id"], x["id"])} if isinstance(x, dict)
                           else id_map.get(x, x) for x in res.get("b_only", [])],
            }
        return g2

    gold2 = remap_ids_in_gold(gold)

    # --- verify before writing ---
    after = collision_pairs(a2["concepts"], b2["concepts"])
    if after != before:
        print("REFUSING TO WRITE: surface-collision structure changed.\n"
              f"  lost:  {sorted(before - after)}\n  gained: {sorted(after - before)}", file=sys.stderr)
        return 1
    def string_values(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from string_values(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from string_values(v)

    blob = " ".join(string_values([a2, b2, ref2])).lower()   # values only, not JSON keys
    hits = [b_ for b_ in BANNED if re.search(b_, blob)]
    if hits:
        print(f"REFUSING TO WRITE: banned standard tokens survive in surface: {hits}", file=sys.stderr)
        return 1

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "model_a_kelvin.json").write_text(json.dumps(a2, indent=2))
    (DST / "model_b_rowan.json").write_text(json.dumps(b2, indent=2))
    (DST / "reference.json").write_text(json.dumps(ref2, indent=2))
    (DST / "gold.json").write_text(json.dumps(gold2, indent=2))
    print(f"wrote {DST.relative_to(HERE.parent)}/  (structurally identical to config_tapi_teas; "
          f"{len(before)} surface collisions preserved; no banned tokens)")
    print("Next: python benchmark/pack.py validate, then run the agent on both cases and compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
