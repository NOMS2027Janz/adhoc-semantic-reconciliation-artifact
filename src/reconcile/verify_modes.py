"""Verification modes over the seeded verify_hard set.

Three modes, each returning a verdict per proposal — "pass", "fail", or "referred"
(cannot decide at this placement, so refer onward):

  * byte_round_trip — the naive baseline the MAGIC paper warns against. A forward-and-
    reverse translation compared for surface equality; it cannot see meaning, so it passes
    any type-compatible pair, including a wrong one that round-trips. Deterministic.
  * invariant_round_trip — an LLM verifier that judges whether the correspondence preserves
    the semantic invariants, reading the two sides' static records. It catches wrong pairs
    whose records differ in an invariant-relevant way (meaning-visible), but is blind to a
    byte-clean wrong pair whose records are identical. Capability-dependent.
  * virtual_operation — the correctness-by-construction step: exercise the correspondence on
    the graph. For services, a provision-and-read-back against the invariants (the instance
    Oracle); for devices/sections, interrogate an authoritative fact on BOTH sides and
    compare. It catches the byte-clean errors the other modes miss — where a live side is
    available. Its reach falls as cognition recedes (an inert side cannot be interrogated).

Deterministic modes need no model; invariant_round_trip calls one. The oracle reads ground
truth; the LLM sees only the records in the proposal.
"""
from __future__ import annotations

import json
import time

from pydantic import BaseModel

from reconcile.instance import Oracle, SVC_TYPE

PASS, FAIL, REFER = "pass", "fail", "referred"


# ---- byte round-trip (deterministic) -----------------------------------------------------
def byte_round_trip(proposal: dict) -> str:
    # a byte round-trip passes whenever forward-then-reverse reproduces the surface form;
    # for a type-compatible pair it always does, so it cannot fail a wrong-but-round-tripping
    # correspondence. This is exactly the insufficiency the paper names.
    return PASS if proposal.get("byte_roundtrips", True) else FAIL


# ---- virtual operation (deterministic, via the instance Oracle) --------------------------
def virtual_operation(proposal: dict, oracle: Oracle) -> str:
    a_id, b_id = proposal["a_id"], proposal["b_id"]
    ta = proposal["a"]["type"]
    tb = proposal["b"]["type"]
    if ta == SVC_TYPE and tb == SVC_TYPE:
        r = oracle.virtual_provision(a_id, b_id)
        if not r.ok:
            return REFER                       # no live side to provision on
        return PASS if r.answer.get("confirmed") else FAIL
    # device/section: interrogate an authoritative fact on both sides and compare
    ra = oracle.interrogate(a_id)
    rb = oracle.interrogate(b_id)
    if not ra.ok or not rb.ok:
        return REFER                           # a side is inert / no budget: cannot compare
    if not ra.answer and not rb.answer:
        return REFER                           # no distinguishing fact to interrogate
    return PASS if ra.answer == rb.answer else FAIL


# ---- invariant round-trip (LLM) ----------------------------------------------------------
VERIFY_SYSTEM = (
    "You VERIFY proposed correspondences between individuals across two graphs of one "
    "network. For each proposal you are given the two individuals' static records (type, "
    "name, attributes, topology). Decide whether treating them as the SAME entity would "
    "preserve the semantic invariants a correct correspondence must hold — endpoint identity, "
    "connectivity, capacity, layer relationships, switching constraints, multiplexing. Judge "
    "by a worked round-trip on those invariants using the records shown: if the two records "
    "imply the same roles and quantities, PASS; if any invariant-relevant detail differs (a "
    "different capacity, different endpoints, a different topology), FAIL. Do not assume two "
    "records with the same name are the same entity, and do not fail two records merely for "
    "different local names. If the two records are IDENTICAL in everything but their local id, "
    "you have no invariant-relevant evidence to fail on, so PASS. Return a verdict for every "
    "proposal."
)

VERIFY_NOTE = {
    "both_cognitive": "\n\nBoth sides are live; you may reason freely from the records.",
    "one_inert": "\n\nOne side is inert (a static snapshot); judge from the records as given.",
    "both_inert": "\n\nBoth sides are static snapshots; judge only from the records as given, "
                  "and do not invent detail beyond them.",
}


class _V(BaseModel):
    id: str
    passes: bool
    reason: str


class _VList(BaseModel):
    verdicts: list[_V]


def invariant_round_trip(proposals: list[dict], placement: str, model: str,
                         invariants: list[str], client=None) -> tuple[dict, dict]:
    """Return {proposal_id: 'pass'|'fail'} and an effort record. One model call for all."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    payload = {
        "invariants": invariants,
        "proposals": [{"id": p["id"], "a": p["a"], "b": p["b"]} for p in proposals],
    }
    system = VERIFY_SYSTEM + VERIFY_NOTE.get(placement, "")
    t0 = time.time()
    from reconcile.compat import parse_compat
    completion = parse_compat(
        client, model,
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(payload, indent=2)}],
        _VList,
    )
    elapsed = time.time() - t0
    result = completion.choices[0].message.parsed
    verdicts = {v.id: (PASS if v.passes else FAIL) for v in result.verdicts}
    # any proposal the model failed to return a verdict for is referred
    for p in proposals:
        verdicts.setdefault(p["id"], REFER)
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    effort = {
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        "latency_s": round(elapsed, 2),
    }
    return verdicts, effort
