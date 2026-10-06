"""Instance-level reconciliation: the populated A-box, and the live-cognition oracle.

This module carries the pieces the schema harness does not: a loader for the
instance case (two A-boxes of individuals, the derived gold, the traps/oracle file,
and the instance/invariant references), and the **oracle** — the deterministic
mechanism that gives a live side its experimental power. The oracle answers in two
ways, matching the MAGIC paper:

  * ``interrogate(individual_id, attribute)`` — one authoritative fact about an
    individual on a LIVE side (a serial, a fibre-id). This is evidence-gathering; it
    is what separates two structurally identical individuals.
  * ``virtual_provision(a_id, b_id)`` — exercise a proposed correspondence by a
    virtual provision-and-read-back, and confirm/refute it *in place* against the
    semantic invariants. This is the correctness-by-construction step; it is only
    meaningful for provisionable (service) individuals.

Availability follows the cognition placement, and every call is counted. The oracle
reads the hidden truth and the invariant signatures — the ground the agent may not
see — so it must be constructed from the case, never exposed to the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SVC_TYPE = "connection-service"


def load_instance_case(case_dir: str | Path) -> "InstanceCase":
    p = Path(case_dir)
    a = json.loads((p / "individuals_a.json").read_text())["individuals"]
    b = json.loads((p / "individuals_b.json").read_text())["individuals"]
    gold = json.loads((p / "instance_gold.json").read_text())
    traps = json.loads((p / "instance_traps.json").read_text())
    reference = json.loads((p / "instance_reference.json").read_text())
    return InstanceCase(name=p.name, a=a, b=b, gold=gold, traps=traps, reference=reference)


@dataclass
class InstanceCase:
    name: str
    a: list[dict]
    b: list[dict]
    gold: dict
    traps: dict
    reference: dict

    @property
    def a_by_id(self):
        return {i["id"]: i for i in self.a}

    @property
    def b_by_id(self):
        return {i["id"]: i for i in self.b}

    @property
    def truth(self) -> dict:
        return {i["id"]: i["_truth"] for i in self.a + self.b}

    @property
    def correct_pairs(self) -> set[frozenset]:
        return {frozenset((c["a"], c["b"])) for c in self.gold["correspondences"]}

    @property
    def false_cognate_pairs(self) -> set[frozenset]:
        return {frozenset((c["a"], c["b"])) for c in self.gold.get("false_cognates", [])}

    @property
    def experiment_only_pairs(self) -> set[frozenset]:
        """The correspondences flagged experiment-only, as id pairs."""
        eo = set(self.gold.get("experiment_only", []))
        a_of = {i["_truth"]: i["id"] for i in self.a}
        b_of = {i["_truth"]: i["id"] for i in self.b}
        return {frozenset((a_of[t], b_of[t])) for t in eo if t in a_of and t in b_of}


@dataclass
class OracleResult:
    ok: bool
    kind: str              # "interrogate" | "provision"
    answer: dict = field(default_factory=dict)
    message: str = ""


class Oracle:
    """Deterministic live-cognition oracle over an instance case.

    placement selects which sides are live and thus interrogable/manipulable:
      both_cognitive -> both sides live; one_inert -> only the non-inert side live;
      both_inert -> nothing live (the oracle refuses every call).
    ``budget`` caps the total number of calls (interrogate + provision); further calls
    are refused so a run cannot loop forever. Counts and confirmed pairs are recorded.
    """

    def __init__(self, case: InstanceCase, placement: str, inert_side: str = "b",
                 budget: int | None = None):
        self.case = case
        self.placement = placement
        self.inert_side = inert_side
        self.budget = budget
        self.truth = case.truth
        self.invariants = case.traps.get("oracle", {}).get("invariants", {})   # by truth
        self.answers = case.traps.get("oracle", {}).get("interrogate", {})       # by individual id
        self.a_ids = set(case.a_by_id)
        self.b_ids = set(case.b_by_id)
        self.type_of = {i["id"]: i["type"] for i in case.a + case.b}
        self.calls = {"interrogate": 0, "provision": 0}
        self.confirmed_pairs: set[frozenset] = set()
        self.refuted_pairs: set[frozenset] = set()
        self.interrogated: set[str] = set()
        self.log: list[dict] = []

    # -- availability ------------------------------------------------------------------
    def _live_side(self, side: str) -> bool:
        if self.placement == "both_cognitive":
            return True
        if self.placement == "both_inert":
            return False
        if self.placement == "one_inert":
            return side != self.inert_side
        return False

    def _side_of(self, ind_id: str) -> str | None:
        if ind_id in self.a_ids:
            return "a"
        if ind_id in self.b_ids:
            return "b"
        return None

    def _budget_left(self) -> bool:
        if self.budget is None:
            return True
        return (self.calls["interrogate"] + self.calls["provision"]) < self.budget

    # -- mechanisms --------------------------------------------------------------------
    def interrogate(self, ind_id: str, attribute: str | None = None) -> OracleResult:
        if not self._budget_left():
            return OracleResult(False, "interrogate", message="oracle budget exhausted")
        side = self._side_of(ind_id)
        if side is None:
            return OracleResult(False, "interrogate", message=f"no such individual '{ind_id}'")
        if not self._live_side(side):
            return OracleResult(False, "interrogate",
                                message=f"individual '{ind_id}' is on an inert side and cannot be interrogated")
        self.calls["interrogate"] += 1
        self.interrogated.add(ind_id)
        facts = self.answers.get(ind_id, {})
        if attribute:
            val = facts.get(attribute)
            if val is not None:
                answer, msg = {attribute: val}, ""
            else:
                available = sorted(facts.keys())
                hint = f"; interrogable attributes: {', '.join(available)}" if available else ""
                answer, msg = {}, f"no authoritative '{attribute}' for '{ind_id}'{hint}"
        else:
            answer, msg = dict(facts), ("" if facts else f"nothing further to interrogate on '{ind_id}'")
        self.log.append({"call": "interrogate", "id": ind_id, "attribute": attribute, "answer": answer})
        return OracleResult(True, "interrogate", answer=answer, message=msg)

    def virtual_provision(self, a_id: str, b_id: str) -> OracleResult:
        """Provision a service through the candidate co-reference and read the objects
        back, confirming or refuting against the semantic invariants."""
        if not self._budget_left():
            return OracleResult(False, "provision", message="oracle budget exhausted")
        if a_id not in self.a_ids or b_id not in self.b_ids:
            return OracleResult(False, "provision",
                                message="virtual_provision needs one id from each side (a_id from A, b_id from B)")
        # provisioning happens on a live side; needs at least the live side present
        if self.placement == "both_inert":
            return OracleResult(False, "provision",
                                message="both sides inert: no live system to provision on")
        if self.placement == "one_inert" and not (self._live_side("a") or self._live_side("b")):
            return OracleResult(False, "provision", message="no live side to provision on")
        if self.type_of.get(a_id) != SVC_TYPE or self.type_of.get(b_id) != SVC_TYPE:
            return OracleResult(False, "provision",
                                message="only services can be provisioned; interrogate devices/sections instead")
        self.calls["provision"] += 1
        ta, tb = self.truth.get(a_id), self.truth.get(b_id)
        inv_a, inv_b = self.invariants.get(ta, {}), self.invariants.get(tb, {})
        diverging = sorted(k for k in set(inv_a) | set(inv_b) if inv_a.get(k) != inv_b.get(k))
        pair = frozenset((a_id, b_id))
        if ta == tb and not diverging:
            self.confirmed_pairs.add(pair)
            res = OracleResult(True, "provision", answer={"confirmed": True, "invariants_preserved": True},
                               message="all invariants preserved in place")
        else:
            self.refuted_pairs.add(pair)
            detail = {k: {"A": inv_a.get(k), "B": inv_b.get(k)} for k in diverging}
            res = OracleResult(True, "provision",
                               answer={"confirmed": False, "diverging_invariants": detail},
                               message="invariants diverge: the two are not the same service")
        self.log.append({"call": "virtual_provision", "a_id": a_id, "b_id": b_id, "answer": res.answer})
        return res
