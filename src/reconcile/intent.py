"""Intent-level reconciliation: the declarative demand, the concrete realisation, and
the deterministic live-cognition oracle for the second operational setting.

Where the first setting reconciled two *structural* models by an equivalence, this one
reconciles a declarative **intent** against a concrete **realisation** by *refinement*:
a bound (bandwidth >=, latency <=, availability >=, protection required) is not equal to
a realisation, it is *satisfied by* one. This module carries the domain predicates that
make satisfaction, feasibility, best-achievable negotiation, and fulfilment deterministic
and testable; the case loader; and the **IntentOracle** — the live-cognition mechanism a
live side wields, gated by the cognition placement, exactly as the instance study's oracle.

The oracle answers four ways:

  * ``check_feasibility(intent, realisation)`` — the provider, live, reports which bounds
    a realisation *actually* satisfies right now, from the hidden operational truth (which
    can deviate from the advertised catalogue). This is what separates an experiment-only
    intent — one whose advertised attributes give the wrong satisfaction verdict — from a
    statically-resolvable one.
  * ``best_achievable(intent)`` — the provider, live, computes the best-achievable offer
    when no realisation fully satisfies: the realisation that degrades only the lowest
    priority bounds, per the priority order in force.
  * ``consult_policy(intent, realisation)`` — the consumer's **movable policy** renders
    accept / reject against its hard bounds, affordability floor, and flow-class rule.
    Available where consumer judgement is present — a live consumer *or* a pre-placed
    policy — and withheld from a mute description (which must refer the decision onward).
  * ``read_operational(intent, hop)`` — live telemetry: the operational reading of the
    in-service realisation at a lifecycle hop, for assure-up.

Every call is counted; availability follows the placement. The oracle reads the hidden
operational truth and the policies, so it is constructed from the case and never exposed
to the prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Domain predicates (pure; shared by the oracle, the case builder, and the gold deriver)
# ---------------------------------------------------------------------------------------

# discrete ODU client-signal rates -> approximate usable bandwidth (Gbit/s), OTN-grounded
ODU_BW = {"ODU0": 1.25, "ODU1": 2.5, "ODU2": 10.0, "ODU3": 40.0, "ODU4": 100.0}

BOUND_KINDS = ("bandwidth", "latency", "availability", "protection")


def violated_bounds(bounds: dict, attrs: dict) -> set[str]:
    """The set of bound kinds an offer's attributes VIOLATE. A bound left None/absent in
    the intent is unconstrained. ``attrs`` carries bw_gbps, latency_ms, availability, and
    protection ('none' | '1+1')."""
    v: set[str] = set()
    bw_min = bounds.get("bandwidth_min")
    if bw_min is not None and attrs.get("bw_gbps", 0) < bw_min:
        v.add("bandwidth")
    lat_max = bounds.get("latency_max")
    if lat_max is not None and attrs.get("latency_ms", 1e9) > lat_max:
        v.add("latency")
    av_min = bounds.get("availability_min")
    if av_min is not None and attrs.get("availability", 0) < av_min:
        v.add("availability")
    if bounds.get("protection_required") and attrs.get("protection", "none") == "none":
        v.add("protection")
    return v


def satisfies_all(bounds: dict, attrs: dict) -> bool:
    return not violated_bounds(bounds, attrs)


def best_achievable(bounds: dict, options: list[dict], priority: list[str]) -> dict | None:
    """The best-achievable offer among ``options`` (each {id, attrs}) under a priority order
    (most-important bound kind first). The rule: prefer the option whose most-important
    violated bound is as LOW priority as possible (so it degrades only what matters least),
    then fewer violations, then lower cost. A fully-satisfying option wins outright.
    Deterministic; ties broken by id for stability."""
    if not options:
        return None
    rank = {k: i for i, k in enumerate(priority)}
    unranked = len(priority)          # a bound not in the priority list ranks last
    none_rank = len(priority) + 1     # 'no violation' beats every violation

    def worst_rank(v: set[str]) -> int:
        if not v:
            return none_rank
        return min(rank.get(k, unranked) for k in v)

    def key(opt):
        v = violated_bounds(bounds, opt["attrs"])
        # maximize worst_rank (degrade least-important), minimize count, minimize cost
        return (-worst_rank(v), len(v), opt["attrs"].get("cost", 0), opt["id"])

    return sorted(options, key=key)[0]


def policy_decision(bounds: dict, offer_attrs: dict, policy: dict) -> str:
    """The movable policy's verdict on an offer: 'accept' or 'reject' (reject == refer
    onward). Reject if the offer violates any of the policy's HARD bounds, or exceeds the
    affordability floor; otherwise accept (a soft-degraded but affordable offer is taken)."""
    hard = set(policy.get("hard_bounds", []))
    if violated_bounds(bounds, offer_attrs) & hard:
        return "reject"
    floor = policy.get("affordability_floor")
    if floor is not None and offer_attrs.get("cost", 0) > floor:
        return "reject"
    return "accept"


def fulfilment_status(bounds: dict, reading: dict, margin: float = 0.10) -> str:
    """Assure-up: map an operational reading to met / at_risk / breach. A breach is any
    violated bound; at_risk is a satisfied reading sitting within ``margin`` of a threshold
    on the tight side; otherwise met."""
    if violated_bounds(bounds, reading):
        return "breach"
    at_risk = False
    bw_min = bounds.get("bandwidth_min")
    if bw_min is not None and reading.get("bw_gbps", 0) < bw_min * (1 + margin):
        at_risk = True
    lat_max = bounds.get("latency_max")
    if lat_max is not None and reading.get("latency_ms", 0) > lat_max * (1 - margin):
        at_risk = True
    av_min = bounds.get("availability_min")
    if av_min is not None and reading.get("availability", 0) < 1 - (1 - av_min) * (1 - margin):
        at_risk = True
    return "at_risk" if at_risk else "met"


def remediation_class(bounds: dict, options: list[dict], policy: dict) -> str:
    """On a breach: 'self_remediable' if some alternative option the policy would ACCEPT
    exists; else 'needs_consumer'."""
    for opt in options:
        if policy_decision(bounds, opt["attrs"], policy) == "accept":
            return "self_remediable"
    return "needs_consumer"


# ---------------------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------------------

def load_intent_case(case_dir: str | Path) -> "IntentCase":
    p = Path(case_dir)

    def j(name):
        return json.loads((p / name).read_text())

    return IntentCase(
        name=p.name,
        intents=j("intents.json")["intents"],
        realisations=j("catalogue.json")["realisations"],
        policies=j("policies.json")["policies"],
        lifecycle=j("lifecycle.json")["trajectories"],
        traps=j("intent_traps.json"),
        reference=j("intent_reference.json"),
        gold=j("intent_gold.json") if (p / "intent_gold.json").exists() else {},
    )


@dataclass
class IntentCase:
    name: str
    intents: list[dict]
    realisations: list[dict]
    policies: list[dict]
    lifecycle: list[dict]
    traps: dict
    reference: dict
    gold: dict = field(default_factory=dict)

    @property
    def intent_by_id(self):
        return {i["id"]: i for i in self.intents}

    @property
    def realisation_by_id(self):
        return {r["id"]: r for r in self.realisations}

    @property
    def policy_by_id(self):
        return {p["id"]: p for p in self.policies}

    @property
    def actual_attrs(self) -> dict:
        """Hidden operational truth: realisation_id -> actual attribute vector."""
        return self.traps.get("actual_attrs", {})

    def options_for(self, pair: str, actual: bool = True) -> list[dict]:
        """Realisations for an endpoint pair, as {id, attrs}. actual=True uses the hidden
        operational truth (the oracle's view); actual=False uses advertised (the catalogue)."""
        out = []
        for r in self.realisations:
            if r.get("pair") != pair:
                continue
            attrs = self.actual_attrs.get(r["id"]) if actual else r.get("advertised")
            if attrs:
                out.append({"id": r["id"], "attrs": attrs})
        return out


# ---------------------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------------------

@dataclass
class OracleResult:
    ok: bool
    kind: str
    answer: dict = field(default_factory=dict)
    message: str = ""


class IntentOracle:
    """Deterministic live-cognition oracle over an intent case.

    Placement gates which sides are live:
      both_cognitive -> provider live, consumer judgement present;
      one_inert      -> the ``inert_side`` ('o' consumer | 'n' provider) is a static
                        description; when the consumer is inert, ``consumer_mode`` says
                        whether a pre-placed 'policy' still decides or it is 'mute';
      both_inert     -> nothing live; feasibility falls back to advertised, no policy.

    ``policy`` is the movable policy in force (priority order + hard bounds + affordability);
    ``budget`` caps total calls. Counts and a call log are recorded.
    """

    def __init__(self, case: IntentCase, placement: str, policy: dict | None = None,
                 inert_side: str = "n", consumer_mode: str = "policy",
                 budget: int | None = None):
        self.case = case
        self.placement = placement
        self.policy = policy or {}
        self.inert_side = inert_side
        self.consumer_mode = consumer_mode
        self.budget = budget
        self.margin = case.traps.get("margin", 0.10)
        self.operational = case.traps.get("operational", {})
        self.intent_by_id = case.intent_by_id
        self.calls = {"feasibility": 0, "best_achievable": 0, "policy": 0, "telemetry": 0}
        self.log: list[dict] = []

        self.provider_live = (placement == "both_cognitive"
                              or (placement == "one_inert" and inert_side != "n"))
        if placement == "both_cognitive":
            self.policy_available = True
        elif placement == "both_inert":
            self.policy_available = False
        elif inert_side == "o":                       # consumer is the inert side
            self.policy_available = (consumer_mode == "policy")
        else:                                          # provider inert, consumer live
            self.policy_available = True

    # -- budget --------------------------------------------------------------------------
    def _budget_left(self) -> bool:
        if self.budget is None:
            return True
        return sum(self.calls.values()) < self.budget

    def _pair_of(self, intent_id: str) -> str | None:
        it = self.intent_by_id.get(intent_id)
        return it.get("pair") if it else None

    # -- mechanisms ----------------------------------------------------------------------
    def check_feasibility(self, intent_id: str, realisation_id: str) -> OracleResult:
        if not self._budget_left():
            return OracleResult(False, "feasibility", message="oracle budget exhausted")
        if not self.provider_live:
            return OracleResult(False, "feasibility",
                                message="provider is inert: no live feasibility check; use the "
                                        "advertised catalogue")
        it = self.intent_by_id.get(intent_id)
        if not it:
            return OracleResult(False, "feasibility", message=f"no such intent '{intent_id}'")
        attrs = self.case.actual_attrs.get(realisation_id)
        if not attrs:
            return OracleResult(False, "feasibility",
                                message=f"no such realisation '{realisation_id}'")
        self.calls["feasibility"] += 1
        v = sorted(violated_bounds(it["bounds"], attrs))
        ans = {"realisation": realisation_id, "satisfies_all": not v, "violated_bounds": v,
               "attrs": attrs}
        self.log.append({"call": "check_feasibility", "intent": intent_id,
                         "realisation": realisation_id, "answer": ans})
        return OracleResult(True, "feasibility", answer=ans)

    def best_achievable(self, intent_id: str) -> OracleResult:
        if not self._budget_left():
            return OracleResult(False, "best_achievable", message="oracle budget exhausted")
        if not self.provider_live:
            return OracleResult(False, "best_achievable",
                                message="provider is inert: cannot compute a fresh best-achievable "
                                        "offer")
        it = self.intent_by_id.get(intent_id)
        if not it:
            return OracleResult(False, "best_achievable", message=f"no such intent '{intent_id}'")
        priority = self.policy.get("priority", list(BOUND_KINDS))
        options = self.case.options_for(it["pair"], actual=True)
        best = best_achievable(it["bounds"], options, priority)
        if not best:
            return OracleResult(False, "best_achievable",
                                message=f"no realisations catalogued for pair '{it['pair']}'")
        self.calls["best_achievable"] += 1
        v = sorted(violated_bounds(it["bounds"], best["attrs"]))
        ans = {"offer": best["id"], "attrs": best["attrs"], "violated_bounds": v,
               "satisfies_all": not v, "priority_used": priority}
        self.log.append({"call": "best_achievable", "intent": intent_id, "answer": ans})
        return OracleResult(True, "best_achievable", answer=ans)

    def consult_policy(self, intent_id: str, realisation_id: str) -> OracleResult:
        if not self._budget_left():
            return OracleResult(False, "policy", message="oracle budget exhausted")
        if not self.policy_available:
            return OracleResult(False, "policy",
                                message="no consumer judgement available (mute description): refer "
                                        "the decision onward")
        it = self.intent_by_id.get(intent_id)
        if not it:
            return OracleResult(False, "policy", message=f"no such intent '{intent_id}'")
        attrs = self.case.actual_attrs.get(realisation_id)
        if not attrs:
            return OracleResult(False, "policy", message=f"no such realisation '{realisation_id}'")
        self.calls["policy"] += 1
        decision = policy_decision(it["bounds"], attrs, self.policy)
        ans = {"realisation": realisation_id, "decision": decision,
               "hard_bounds": self.policy.get("hard_bounds", []),
               "affordability_floor": self.policy.get("affordability_floor"),
               "violated_bounds": sorted(violated_bounds(it["bounds"], attrs))}
        self.log.append({"call": "consult_policy", "intent": intent_id,
                         "realisation": realisation_id, "answer": ans})
        return OracleResult(True, "policy", answer=ans)

    def read_operational(self, trajectory_id: str, hop_id: str) -> OracleResult:
        if not self._budget_left():
            return OracleResult(False, "telemetry", message="oracle budget exhausted")
        if not self.provider_live:
            return OracleResult(False, "telemetry",
                                message="provider is inert: no live telemetry")
        key = f"{trajectory_id}/{hop_id}"
        rec = self.operational.get(key)
        if not rec:
            return OracleResult(False, "telemetry", message=f"no operational reading for '{key}'")
        self.calls["telemetry"] += 1
        self.log.append({"call": "read_operational", "hop": key, "answer": rec})
        return OracleResult(True, "telemetry", answer=rec)
