#!/usr/bin/env python3
"""Build an INDEPENDENT-LEXICON reconciliation case from the shared-reference evpn case.

E1 (Brad review, point 1). The main benchmark gives both sides the SAME reference entry
ids, so a shared reference resolves correspondence by identity of ids -- the mechanical
shortcut the reference-reconciler control exploits. The reference-lexicons draft names the
HARDER problem as aligning two *independently developed* lexicons, which the benchmark has
not tested.

This builder re-represents `config_evpn` (same two models, same true correspondences) with
two INDEPENDENTLY authored lexicons: MEF-flavoured for side A (ids `la.*`), IETF/EVPN-
flavoured for side B (ids `lb.*`). No id is shared across sides, so the reference-reconciler
control collapses (it can only match equal ids); a cognitive agent must instead ALIGN the
two lexicons by their definitions. The gold is carried unchanged from config_evpn -- the
underlying concepts and their true correspondences do not change, only how the reference
represents them.

Output: benchmark/cases/config_evpn_indeplex/  (a normal schema case whose reference.json is
the union of the two lexicons, marked kind="independent"; reference_a.json / reference_b.json
are also written for readability).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "cases" / "config_evpn"
DST = HERE / "cases" / "config_evpn_indeplex"

# Lexicon A -- MEF carrier-Ethernet vocabulary (side A / model_a_mef).
LA_ENTRIES = [
    {"id": "la.evc", "label": "Ethernet Virtual Connection (EVC)", "synonyms": ["E-LAN service", "carrier-Ethernet service instance"], "class": "service", "definition": "The customer's carrier-Ethernet service as one virtual-connection instance (a MEF EVC).", "example": "the ACME multipoint E-LAN service"},
    {"id": "la.uni", "label": "User-Network Interface (UNI)", "synonyms": ["subscriber attachment", "access interface"], "class": "access", "definition": "The interface at which a subscriber site attaches to the carrier-Ethernet network.", "example": "the ACME headquarters UNI"},
    {"id": "la.evcep", "label": "EVC endpoint", "synonyms": ["EVC per-UNI endpoint"], "class": "endpoint", "definition": "The endpoint of the EVC at one UNI.", "example": "the EVC endpoint at the headquarters UNI"},
    {"id": "la.bcast", "label": "customer broadcast segment", "synonyms": ["VLAN broadcast domain", "network-segment"], "class": "l2-domain", "definition": "The customer VLAN carried by the service as a single layer-2 broadcast domain -- a forwarding domain, not a multi-homing link bundle.", "example": "the customer VLAN 200 broadcast domain"},
    {"id": "la.bwp", "label": "Bandwidth Profile", "synonyms": ["ingress bandwidth profile", "CIR/EIR profile"], "class": "rate", "definition": "The committed/excess ingress rate policed at a UNI (a MEF bandwidth profile).", "example": "CIR 200 Mbit/s, EIR 50 Mbit/s"},
    {"id": "la.cos", "label": "Class of Service (CoS)", "synonyms": ["CoS", "service class"], "class": "cos", "definition": "The forwarding-treatment class applied to service frames.", "example": "the real-time class mapped from PCP 5"},
    {"id": "la.macl", "label": "source-MAC learning", "synonyms": ["customer MAC learning"], "class": "l2-control", "definition": "Learning customer source-MAC addresses at a UNI to populate the forwarding table; distinct from handling a MAC that relocates.", "example": "dynamic learning at the headquarters UNI"},
    {"id": "la.mtu", "label": "service MTU", "synonyms": ["MEF service MTU"], "class": "framing", "definition": "The maximum service frame size carried end to end.", "example": "9100-byte jumbo frames"},
    {"id": "la.avail", "label": "service availability objective", "synonyms": ["availability SLO"], "class": "assurance", "definition": "The service-level availability objective; a service-view construct with no network-model counterpart.", "example": "99.99% monthly availability"},
]

# Lexicon B -- IETF/EVPN vocabulary (side B / model_b_evpn).
LB_ENTRIES = [
    {"id": "lb.evi", "label": "EVPN Instance (EVI)", "synonyms": ["MAC-VRF", "EVPN service instance"], "class": "service", "definition": "The EVPN instance realising one Ethernet VPN service (a MAC-VRF).", "example": "EVI 5000 for the ACME service"},
    {"id": "lb.acif", "label": "Attachment Circuit (AC)", "synonyms": ["AC", "access circuit"], "class": "access", "definition": "The circuit or interface by which a customer site attaches to a provider-edge router.", "example": "the AC on PE1 for the ACME headquarters"},
    {"id": "lb.vna", "label": "VPN network access", "synonyms": ["network access point"], "class": "endpoint", "definition": "The service's access point at one attachment -- the per-AC network access of the VPN.", "example": "the VPN network access at the headquarters site"},
    {"id": "lb.bd", "label": "bridge domain", "synonyms": ["VLAN", "MAC-VRF bridge table"], "class": "l2-domain", "definition": "The layer-2 bridge domain (customer VLAN) over which the EVPN forwards; one broadcast domain.", "example": "the bridge-domain for customer VLAN 200"},
    {"id": "lb.policer", "label": "ingress policer", "synonyms": ["rate limiter", "ingress rate policy"], "class": "rate", "definition": "The ingress committed/excess rate enforced at the attachment.", "example": "policer at CIR 200 Mbit/s, EIR 50 Mbit/s"},
    {"id": "lb.tclass", "label": "traffic class", "synonyms": ["forwarding class", "TC"], "class": "cos", "definition": "The forwarding class applied to frames on the instance.", "example": "the traffic class for the real-time queue"},
    {"id": "lb.macl", "label": "MAC learning", "synonyms": ["MAC-VRF learning", "local MAC learning"], "class": "l2-control", "definition": "Learning local customer source-MAC addresses into the MAC-VRF forwarding table; distinct from EVPN MAC mobility.", "example": "local MAC learning on PE1"},
    {"id": "lb.mtu", "label": "interface MTU", "synonyms": ["L2 MTU"], "class": "framing", "definition": "The maximum frame size carried on the interface end to end.", "example": "a 9100-byte interface MTU"},
    {"id": "lb.eseg", "label": "Ethernet Segment (ES)", "synonyms": ["ESI", "multihoming segment"], "class": "multihoming", "definition": "The set of links by which a multi-homed site attaches to more than one provider-edge router, identified by an ESI; a multi-homing construct, not a broadcast domain.", "example": "the dual-homed site's ESI on PE2 and PE3"},
    {"id": "lb.macmob", "label": "MAC mobility", "synonyms": ["MAC-move", "EVPN MAC mobility"], "class": "l2-control", "definition": "The EVPN mechanism handling a customer MAC that relocates between provider-edge routers, using a sequence number; distinct from local MAC learning.", "example": "a sequence bump when a MAC moves from PE1 to PE2"},
]

# shared reference id -> this side's independent-lexicon id
A_MAP = {"eth-vpn": "la.evc", "uni": "la.uni", "svc-endpoint": "la.evcep", "vlan": "la.bcast",
         "bw-profile": "la.bwp", "cos": "la.cos", "mac-learning": "la.macl", "mtu": "la.mtu",
         "svc-availability": "la.avail"}
B_MAP = {"eth-vpn": "lb.evi", "uni": "lb.acif", "svc-endpoint": "lb.vna", "vlan": "lb.bd",
         "bw-profile": "lb.policer", "cos": "lb.tclass", "mac-learning": "lb.macl", "mtu": "lb.mtu",
         "ethernet-segment": "lb.eseg", "mac-mobility": "lb.macmob"}


def _rebind(model_path: Path, id_map: dict) -> dict:
    d = json.loads(model_path.read_text())
    for c in d["concepts"]:
        r = c.get("ref")
        if r is not None:
            if r not in id_map:
                raise SystemExit(f"unmapped ref {r!r} in {model_path.name}; extend the id map")
            c["ref"] = id_map[r]
    return d


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    # 1. models, rebound to their own lexicon ids
    a = _rebind(SRC / "model_a_mef.json", A_MAP)
    b = _rebind(SRC / "model_b_evpn.json", B_MAP)
    a["note"] = "config_evpn side A, rebound to the independent MEF lexicon (la.*) for E1."
    b["note"] = "config_evpn side B, rebound to the independent EVPN lexicon (lb.*) for E1."
    (DST / "model_a_mef.json").write_text(json.dumps(a, indent=2))
    (DST / "model_b_evpn.json").write_text(json.dumps(b, indent=2))

    # 2. the two lexicons, and their union as the case reference (kind=independent)
    ref_a = {"id": "lex.mef.v1", "kind": "lexical", "note": "Independent MEF-side lexicon (E1).",
             "entries": LA_ENTRIES}
    ref_b = {"id": "lex.evpn.v1", "kind": "lexical", "note": "Independent EVPN-side lexicon (E1).",
             "entries": LB_ENTRIES}
    (DST / "reference_a.json").write_text(json.dumps(ref_a, indent=2))
    (DST / "reference_b.json").write_text(json.dumps(ref_b, indent=2))
    union = {
        "id": "ref.evpn.indeplex.v1", "kind": "independent",
        "note": ("Two INDEPENDENTLY authored lexicons (la.* for the MEF side, lb.* for the "
                 "EVPN side) presented as one payload. No id is shared across sides: the "
                 "reference-reconciler control cannot match, and a cognitive agent must align "
                 "the two lexicons by their definitions. Cross-lexicon false cognates are "
                 "planted (la.bcast 'segment' vs lb.eseg 'Ethernet Segment'; la.macl "
                 "'source-MAC learning' vs lb.macmob 'MAC mobility')."),
        "entries": LA_ENTRIES + LB_ENTRIES,
    }
    (DST / "reference.json").write_text(json.dumps(union, indent=2))

    # 3. gold + traps carried unchanged from config_evpn (same models, same true pairs)
    gold = json.loads((SRC / "gold.json").read_text())
    gold["case"] = "config_evpn_indeplex"
    gold["seed"] = ("carried from config_evpn: same two models and same true correspondences, "
                    "re-represented with two independent lexicons (E1). The per-correspondence "
                    "'ref' fields name the old shared entry and are informational only; scoring "
                    "uses the (a,b) pairs.")
    gold["note"] = "True correspondences are a property of the concepts, not of the reference; carried from config_evpn. Do not hand-edit."
    (DST / "gold.json").write_text(json.dumps(gold, indent=2))
    (DST / "traps.json").write_text((SRC / "traps.json").read_text())

    print(f"wrote {DST.relative_to(HERE.parent)}/ : "
          f"{len(LA_ENTRIES)} la + {len(LB_ENTRIES)} lb entries, "
          f"{len(gold['correspondences'])} gold pairs, {len(gold.get('false_cognates', []))} traps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
