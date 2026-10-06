#!/usr/bin/env python3
"""Build E11: a structurally MESSY vendor-YANG case paired with a clean standard model (Brad pt 15).

The tidy cases give both sides clean labels and good glosses. Real vendor YANG is messy: deep nested
paths, concepts spread across augmenting modules, a config/state split (the same thing appears twice),
sparse or inherited descriptions, and leafref indirection. This case pairs a clean IETF-style L3VPN
service model (side A) with an "Acme" vendor device model of the same VPN (side B) that carries every
one of those difficulties, so we can ask whether reconciliation survives real vendor structural
complexity rather than the tidy L3VPN case.

Messiness encoded on the vendor side:
  * deep-path labels (acme-l3vpn:instances/instance/vrf/...) instead of clean terms;
  * SPARSE glosses (empty or one-word) on several vendor concepts -> must reconcile by structure;
  * a CONFIG/STATE split: the route-distinguisher appears as both a config leaf and an operational
    state leaf (two vendor concepts) that both correspond to ONE standard concept (2:1);
  * CROSS-MODULE spread: the attachment and the BGP neighbour come from augmenting modules
    (acme-if:, acme-bgp:), not the main l3vpn module;
  * a LEAFREF-INDIRECTION false cognate: acme's vrf/import-policy (a routing-policy filter) surface-
    resembles the standard import-route-target by the 'import' token, but denotes a different thing;
  * NATIVE GAPS both ways: a standard service-bandwidth with no vendor counterpart, and vendor-only
    knobs (segment-id, and the import-policy whose true standard partner is absent).

Writes benchmark/cases/config_vendor_messy/ (a normal schema case). Gold is authored here and
validated by controls offline: the label-matcher should do POORLY (deep vendor paths do not surface-
match standard terms), which is the point; the reference-reconciler should resolve it (shared binding).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DST = HERE / "cases" / "config_vendor_messy"

# --- standard side (clean IETF-style L3VPN service model) ------------------------------------
STD = [
    {"id": "s.vpnsvc", "label": "vpn-service", "kind": "service", "ref": "vpn-service",
     "synonyms": ["l3vpn", "vpn-instance"], "gloss": "one L3VPN service instance across the network",
     "example": "the ACME L3VPN", "relations": [{"rel": "has-access", "target": "s.vpnaccess"}]},
    {"id": "s.rd", "label": "route-distinguisher", "kind": "identifier", "ref": "route-distinguisher",
     "synonyms": ["RD"], "gloss": "the 8-byte value making VPN routes globally unique",
     "example": "64512:100", "relations": [{"rel": "of", "target": "s.vpnsvc"}]},
    {"id": "s.importrt", "label": "import-route-target", "kind": "policy-value", "ref": "import-rt",
     "synonyms": ["import-RT"], "gloss": "the route-target value on which routes are imported into the VRF",
     "example": "target:64512:100", "relations": [{"rel": "of", "target": "s.vpnsvc"}]},
    {"id": "s.exportrt", "label": "export-route-target", "kind": "policy-value", "ref": "export-rt",
     "synonyms": ["export-RT"], "gloss": "the route-target value attached to routes exported from the VRF",
     "example": "target:64512:100", "relations": [{"rel": "of", "target": "s.vpnsvc"}]},
    {"id": "s.vpnaccess", "label": "vpn-network-access", "kind": "access", "ref": "vpn-access",
     "synonyms": ["site-network-access", "attachment"], "gloss": "the customer attachment to the VPN at a PE",
     "example": "the ACME site attachment on PE1", "relations": [{"rel": "carries", "target": "s.bgpneighbor"}]},
    {"id": "s.bgpneighbor", "label": "pe-ce-bgp-neighbor", "kind": "protocol", "ref": "bgp-neighbor",
     "synonyms": ["bgp-peer"], "gloss": "the PE-CE BGP peering that exchanges customer routes",
     "example": "neighbor 10.0.0.2", "relations": [{"rel": "has-timer", "target": "s.bgpka"}]},
    {"id": "s.bgpka", "label": "bgp-keepalive", "kind": "parameter", "ref": "bgp-keepalive",
     "synonyms": ["keepalive-interval"], "gloss": "the BGP keepalive interval for the PE-CE peer",
     "example": "30 seconds", "relations": []},
    {"id": "s.ipconn", "label": "ip-connection", "kind": "addressing", "ref": "ip-connection",
     "synonyms": ["ip-address", "prefix"], "gloss": "the IP addressing on the customer attachment",
     "example": "10.0.0.1/30", "relations": [{"rel": "on", "target": "s.vpnaccess"}]},
    {"id": "s.svcbw", "label": "service-bandwidth", "kind": "rate", "ref": "service-bandwidth",
     "synonyms": ["svc-bw", "guaranteed-bandwidth"], "gloss": "the contracted service bandwidth (no vendor counterpart)",
     "example": "200 Mbit/s", "relations": [{"rel": "of", "target": "s.vpnsvc"}]},
]

# --- vendor side (messy Acme device model) ---------------------------------------------------
# sparse glosses are deliberately empty or one-word; labels are deep YANG paths; some come from
# augmenting modules (acme-if:, acme-bgp:).
VEN = [
    {"id": "v.inst", "label": "acme-l3vpn:instances/instance", "kind": "container", "ref": "vpn-service",
     "synonyms": [], "gloss": "vrf instance.", "example": "instance ACME-VRF",
     "relations": [{"rel": "contains", "target": "v.rd_cfg"}]},
    {"id": "v.rd_cfg", "label": "acme-l3vpn:instances/instance/vrf/rd", "kind": "leaf", "ref": "route-distinguisher",
     "synonyms": [], "gloss": "", "example": "64512:100", "relations": [{"rel": "under", "target": "v.inst"}]},
    {"id": "v.rd_state", "label": "acme-l3vpn:instances/instance/vrf/state/oper-rd", "kind": "leaf", "ref": "route-distinguisher",
     "synonyms": [], "gloss": "operational rd.", "example": "64512:100",
     "relations": [{"rel": "reflects", "target": "v.rd_cfg"}]},
    {"id": "v.rt_imp", "label": "acme-l3vpn:.../vrf/address-family/ipv4-unicast/route-target/import", "kind": "leaf-list",
     "ref": "import-rt", "synonyms": [], "gloss": "", "example": "target:64512:100",
     "relations": [{"rel": "under", "target": "v.inst"}]},
    {"id": "v.rt_exp", "label": "acme-l3vpn:.../vrf/address-family/ipv4-unicast/route-target/export", "kind": "leaf-list",
     "ref": "export-rt", "synonyms": [], "gloss": "", "example": "target:64512:100",
     "relations": [{"rel": "under", "target": "v.inst"}]},
    {"id": "v.imp_policy", "label": "acme-l3vpn:.../vrf/import-policy", "kind": "leafref", "ref": "import-policy",
     "synonyms": [], "gloss": "policy ref.", "example": "-> policy ACME-IN",
     "relations": [{"rel": "references", "target": "v.seg"}]},
    {"id": "v.attach", "label": "acme-if:interfaces/interface/vrf-attachment", "kind": "augment", "ref": "vpn-access",
     "synonyms": [], "gloss": "", "example": "GE0/1 in ACME-VRF",
     "relations": [{"rel": "binds", "target": "v.inst"}, {"rel": "hosts", "target": "v.bgp_nbr"}]},
    {"id": "v.bgp_nbr", "label": "acme-bgp:bgp/neighbors/neighbor", "kind": "augment", "ref": "bgp-neighbor",
     "synonyms": [], "gloss": "", "example": "neighbor 10.0.0.2",
     "relations": [{"rel": "has", "target": "v.bgp_ka"}]},
    {"id": "v.bgp_ka", "label": "acme-bgp:bgp/neighbors/neighbor/timers/keepalive-interval", "kind": "leaf",
     "ref": "bgp-keepalive", "synonyms": [], "gloss": "", "example": "30",
     "relations": [{"rel": "under", "target": "v.bgp_nbr"}]},
    {"id": "v.ipaddr", "label": "acme-if:interfaces/interface/ipv4/address", "kind": "leaf", "ref": "ip-connection",
     "synonyms": [], "gloss": "", "example": "10.0.0.1/30", "relations": [{"rel": "on", "target": "v.attach"}]},
    {"id": "v.seg", "label": "acme-l3vpn:instances/instance/vrf/segment-id", "kind": "leaf", "ref": "segment",
     "synonyms": [], "gloss": "vendor segment id.", "example": "seg 4001", "relations": [{"rel": "under", "target": "v.inst"}]},
]

# --- reference (shared lexical) --------------------------------------------------------------
REF_ENTRIES = [
    {"id": "vpn-service", "label": "L3VPN service instance", "class": "service",
     "definition": "one L3VPN service / VRF instance across the network", "example": "the ACME L3VPN"},
    {"id": "route-distinguisher", "label": "route distinguisher", "class": "identifier",
     "definition": "the value making a VPN's routes globally unique; one logical value whether read from config or operational state", "example": "64512:100"},
    {"id": "import-rt", "label": "import route target", "class": "policy-value",
     "definition": "the route-target community on which routes are imported into the VRF", "example": "target:64512:100"},
    {"id": "export-rt", "label": "export route target", "class": "policy-value",
     "definition": "the route-target community attached to routes exported from the VRF", "example": "target:64512:100"},
    {"id": "vpn-access", "label": "VPN network access", "class": "access",
     "definition": "the customer attachment to the VPN at a provider edge", "example": "the ACME attachment on PE1"},
    {"id": "bgp-neighbor", "label": "PE-CE BGP neighbor", "class": "protocol",
     "definition": "the PE-CE BGP peering exchanging customer routes", "example": "neighbor 10.0.0.2"},
    {"id": "bgp-keepalive", "label": "BGP keepalive interval", "class": "parameter",
     "definition": "the keepalive interval for the PE-CE BGP peer", "example": "30 seconds"},
    {"id": "ip-connection", "label": "IP connection / addressing", "class": "addressing",
     "definition": "the IP addressing configured on the customer attachment", "example": "10.0.0.1/30"},
    {"id": "service-bandwidth", "label": "service bandwidth", "class": "rate",
     "definition": "the contracted service bandwidth (a service-view construct)", "example": "200 Mbit/s"},
    {"id": "import-policy", "label": "import routing-policy", "class": "policy-ref",
     "definition": "a named routing-policy filter applied on import; NOT a route-target value (a common false cognate)", "example": "policy ACME-IN"},
    {"id": "segment", "label": "vendor segment id", "class": "vendor-knob",
     "definition": "an Acme-specific segment identifier with no standard-model counterpart", "example": "seg 4001"},
]

# --- gold ------------------------------------------------------------------------------------
CORRESPONDENCES = [
    ("s.vpnsvc", "v.inst", "vpn-service"),
    ("s.rd", "v.rd_cfg", "route-distinguisher"),
    ("s.rd", "v.rd_state", "route-distinguisher"),          # config/state split -> 2:1
    ("s.importrt", "v.rt_imp", "import-rt"),
    ("s.exportrt", "v.rt_exp", "export-rt"),
    ("s.vpnaccess", "v.attach", "vpn-access"),
    ("s.bgpneighbor", "v.bgp_nbr", "bgp-neighbor"),
    ("s.bgpka", "v.bgp_ka", "bgp-keepalive"),
    ("s.ipconn", "v.ipaddr", "ip-connection"),
]
# false cognate: standard import-route-target vs vendor import-POLICY (both 'import', different things)
FALSE_COGNATES = [
    {"a": "s.importrt", "b": "v.imp_policy",
     "why": "import-route-target vs import-policy: both carry 'import', but a route-target community vs a named routing-policy filter (a leafref). s.importrt's true partner is v.rt_imp; v.imp_policy's standard partner is a native gap."},
]
# native gaps: standard service-bandwidth (no vendor), vendor segment-id and import-policy (no standard)
RESIDUAL = {"a_native": ["s.svcbw"], "b_native": ["v.seg", "v.imp_policy"]}


def _model(system, dialect, modules, concepts, note):
    for c in concepts:
        c.setdefault("synonyms", []); c.setdefault("relations", []); c.setdefault("instances", [])
    return {"system": system, "dialect": dialect, "modules": modules, "note": note, "concepts": concepts}


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    a = _model("IETF-L3NM", "ietf-l3vpn-svc", ["ietf-l3vpn-svc"], STD,
               "Clean IETF-style L3VPN service model (side A) for the vendor-messiness case (E11).")
    b = _model("Acme-Device", "acme-l3vpn/acme-if/acme-bgp", ["acme-l3vpn", "acme-if", "acme-bgp"], VEN,
               "Messy Acme vendor device model (side B): deep paths, sparse glosses, config/state split, cross-module augments, leafref indirection. E11.")
    (DST / "model_a_standard.json").write_text(json.dumps(a, indent=2))
    (DST / "model_b_vendor.json").write_text(json.dumps(b, indent=2))
    (DST / "reference.json").write_text(json.dumps(
        {"id": "ref.l3vpn.vendor.v1", "kind": "lexical",
         "note": "Shared L3VPN lexicon binding the clean standard and messy vendor sides (E11).",
         "entries": REF_ENTRIES}, indent=2))
    (DST / "gold.json").write_text(json.dumps({
        "case": "config_vendor_messy",
        "note": "Standard(A) <-> messy vendor(B) L3VPN correspondences. Includes a config/state 2:1 (s.rd ~ v.rd_cfg AND v.rd_state) and native gaps. Authored + validated by build_vendor_messy_case.py; do not hand-edit.",
        "correspondences": [{"a": a_, "b": b_, "ref": r} for (a_, b_, r) in CORRESPONDENCES],
        "false_cognates": FALSE_COGNATES,
        "residual": RESIDUAL,
    }, indent=2))
    (DST / "traps.json").write_text(json.dumps({"false_cognates": FALSE_COGNATES}, indent=2))
    print(f"wrote {DST.relative_to(HERE.parent)}/ : {len(STD)} standard + {len(VEN)} vendor concepts, "
          f"{len(CORRESPONDENCES)} correspondences (incl. 1 config/state 2:1), "
          f"{len(FALSE_COGNATES)} false cognate, gaps a={RESIDUAL['a_native']} b={RESIDUAL['b_native']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
