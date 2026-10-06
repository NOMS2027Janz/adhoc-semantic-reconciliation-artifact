#!/usr/bin/env python3
"""Derive and validate the E8 dependency-map gold from raw inventory (Brad pt 12).

The dependency map is not stated in the case; it FOLLOWS from the inventory by a multi-hop join:

  * resolve each IP link to its (router, port): a direct port, or via a LAG bundle's member port(s);
  * underlies(optical, ip)  <=  optical line and IP link meet at the same (router, port);
  * underlies(ip, service)  <=  the service runs 'over' that IP link.

`derive_underlies(inventory)` is the deterministic oracle that performs this join -- it is both the
GOLD dependency map the agent's derivation is scored against AND the map fed to correlate() to make
the incident gold. This script also validates the intended difficulty is present (the LAG-bundle
indirection resolves, the name-lure otu5b produces NO edge) and refuses to write an inconsistent
gold.

Writes benchmark/cases/derive_deps_config/deps_gold.json:
  {"underlies": {resource: [resources it underlies]}, "incidents": {scenario_id: [...]}}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reconcile.observability import correlate                    # noqa: E402

CDIR = ROOT / "benchmark" / "cases" / "derive_deps_config"


def derive_underlies(inv: dict) -> dict:
    """Deterministic join: inventory facts -> the underlies (dependency) map."""
    bundle_ports = {}
    for b in inv.get("bundles", []):
        bundle_ports[(b["router"], b["id"])] = list(b.get("members", []))

    # each IP link -> set of (router, port) it occupies (direct or via bundle)
    ip_locs = {}
    for link in inv.get("ip_links", []):
        r = link["router"]
        if "port" in link:
            ip_locs[link["id"]] = {(r, link["port"])}
        elif "bundle" in link:
            ip_locs[link["id"]] = {(r, p) for p in bundle_ports.get((r, link["bundle"]), [])}
        else:
            ip_locs[link["id"]] = set()

    # optical line -> (router, port)
    opt_loc = {o["id"]: (o["router"], o["port"]) for o in inv.get("optical_lines", [])}

    underlies: dict[str, list] = {}

    # optical underlies ip when they meet at the same (router, port)
    for oid, loc in opt_loc.items():
        for lid, locs in ip_locs.items():
            if loc in locs:
                underlies.setdefault(oid, []).append(lid)

    # ip underlies service when the service runs over it
    for svc in inv.get("services", []):
        for lid in svc.get("over", []):
            underlies.setdefault(lid, []).append(svc["id"])

    return {k: sorted(set(v)) for k, v in underlies.items()}


def main() -> int:
    inv = json.loads((CDIR / "inventory.json").read_text())
    underlies = derive_underlies(inv)
    errors: list[str] = []

    # intended edges (from the case design)
    expect = {
        "otu2": ["ip-le3"], "otu5": ["ip-le7"], "otu9": ["ip-le9"],
        "ip-le3": ["l3vpn-acme"], "ip-le7": ["evpn-globex"],
    }
    for src, tgts in expect.items():
        if underlies.get(src) != tgts:
            errors.append(f"expected {src} -> {tgts}; derived {underlies.get(src)}")
    # the LAG-bundle indirection must have resolved (otu5 -> ip-le7 via p7)
    if "ip-le7" not in underlies.get("otu5", []):
        errors.append("bundle indirection failed: otu5 should underlie ip-le7 via bnd-7 -> p7")
    # the name lure must produce NO edge
    if "otu5b" in underlies:
        errors.append(f"name lure otu5b should have no dependency edge; got {underlies['otu5b']}")

    # incident gold via correlate() with the GOLD map
    scenarios = json.loads((CDIR / "scenarios.json").read_text())["scenarios"]
    incidents = {}
    for sc in scenarios:
        res = correlate(sc["symptoms"], underlies, window=sc.get("window", 5.0))
        incidents[sc["id"]] = res["incidents"]
        if len({frozenset(i["symptoms"]) for i in res["incidents"]}) != 2:
            errors.append(f"{sc['id']} should form 2 incidents under the gold map; got {len(res['incidents'])}")

    if errors:
        print("DERIVE_DEPS GOLD VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    out = {
        "case": "derive_deps_config",
        "note": "Dependency-map gold DERIVED by derive_deps_gold.py (deterministic join over inventory); incident gold via correlate() with that map. Do not hand-edit.",
        "underlies": underlies,
        "incidents": incidents,
    }
    (CDIR / "deps_gold.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Wrote derive_deps_config/deps_gold.json")
    print("  gold underlies map:")
    for k in sorted(underlies):
        print(f"    {k} -> {underlies[k]}")
    print("  correlation incidents:")
    for sid, incs in incidents.items():
        desc = "; ".join(f"[{'+'.join(i['symptoms'])}->{i['cause']}]" for i in incs)
        print(f"    {sid}: {len(incs)}  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
