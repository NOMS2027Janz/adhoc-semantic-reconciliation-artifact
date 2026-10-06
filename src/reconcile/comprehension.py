"""SCAFFOLD (portability program, E22) -- destined for src/reconcile/comprehension.py.

Comprehensibility of a SINGLE lifted model to a cognitive CONSUMER that did not author it and is not a
reconciliation partner. This is the direct, consumer-relative measure of lift the study currently lacks:
instead of grading a lift through a downstream reconciliation, hand the model to a consumer and measure
whether it can (i) say what each concept means, (ii) use the model for a task, and (iii) FLAG the parts it
cannot safely understand rather than confabulate.

Three pieces:
  * load_comprehension(case_dir)                  -> Comprehension  (the authored gold; see schema below)
  * comprehend(model_payload, gold, consumer_model, client) -> (ConsumerAnswer, effort)
  * judge_meaning(gold, answer, judge_model, client)        -> list[Verdict]   (strong-judge grading)
  * score_comprehension(gold, answer, verdicts)             -> dict            (pure; offline-testable)

DESIGN DECISIONS (do NOT finalise without Chris -- see _shelf/PORTABILITY_program.md §"Open decisions"):
  - the task set (meaning-QA + use + flag) and whether to add a "restate for a naive peer" task;
  - how `ambiguous_ids` are chosen (they define the hazard signal);
  - per-concept vs holistic scoring.
The prompt bodies below are DRAFT and marked as such.

Reuse note: the OpenAI call mirrors reconcile/construct.py and reconcile/stacks/agent_twoagent.py
(client.chat.completions.parse(model, messages, response_format=<pydantic>)); the judge mirrors the E17
semantic judge (faithful / partial / invented).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


# ---- the authored gold -------------------------------------------------------------------
@dataclass
class MeaningQ:
    id: str
    concept_id: str
    question: str
    answer_key: str
    must_not_confuse_with: list[str] = field(default_factory=list)


@dataclass
class UseTask:
    goal: str
    relevant_ids: list[str]
    irrelevant_ids: list[str]
    answer_key: str = ""


@dataclass
class Comprehension:
    model: str                       # which side of the case ("a" or "b")
    meaning_questions: list[MeaningQ]
    use_task: UseTask
    ambiguous_ids: list[str]         # genuinely under-anchored on the bare surface

    @classmethod
    def load(cls, case_dir: str | Path) -> "Comprehension":
        d = json.loads((Path(case_dir) / "comprehension.json").read_text())
        return cls(
            model=d.get("model", "a"),
            meaning_questions=[MeaningQ(**q) for q in d["meaning_questions"]],
            use_task=UseTask(**d["use_task"]),
            ambiguous_ids=list(d.get("ambiguous_ids", [])),
        )


def load_comprehension(case_dir: str | Path) -> Comprehension:
    return Comprehension.load(case_dir)


# ---- structured consumer + judge outputs -------------------------------------------------
class MeaningAnswer(BaseModel):
    qid: str
    answer: str


class ConsumerAnswer(BaseModel):
    meaning: list[MeaningAnswer]
    use_relevant: list[str] = Field(default_factory=list)     # concept ids the consumer judges relevant
    use_rationale: str = ""
    flagged_ambiguous: list[str] = Field(default_factory=list)  # ids the consumer cannot safely understand


class Verdict(BaseModel):
    qid: str
    label: str    # "faithful" | "partial" | "invented"


class JudgeResult(BaseModel):
    verdicts: list[Verdict]


# ---- DRAFT prompts (task set is a decision point) ----------------------------------------
_CONSUMER_SYSTEM = (
    "You are a cognitive consumer of a semantic model you did NOT author. You will be shown ONE model "
    "(its concepts, with whatever meaning and anchoring it carries). Do three things, by MEANING not by "
    "label. (1) For each meaning question, say what the named concept denotes, in terms a domain engineer "
    "would recognise. (2) For the use task, list the concept ids directly relevant to the goal and say "
    "briefly how each is used. (3) List the ids of any concepts you cannot safely understand from what you "
    "were given -- under-specified, ambiguous, or liable to be misread -- rather than guessing. Do not "
    "invent meaning you cannot support from the model; flagging is preferred to confabulation."
)  # DRAFT

_JUDGE_SYSTEM = (
    "You grade a consumer's stated meaning for each concept against an authored answer key. For each "
    "question return a verdict: 'faithful' (matches the key's meaning), 'partial' (right direction, "
    "missing or hedged), or 'invented' (asserts a meaning the key does not support, e.g. confuses it with "
    "a look-alike). Judge meaning, not wording."
)  # DRAFT


def _parse(client, model, system, payload, schema):
    # parse_compat: strict json_schema first, JSON-mode fallback where a provider (DeepSeek, some
    # Ollama builds) rejects the schema form. Lets a cross-family CONSUMER be graded by the same path
    # OpenAI uses, so the judge stays comparable across consumer families.
    from reconcile.compat import parse_compat
    completion = parse_compat(
        client, model,
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(payload, indent=2)}],
        schema,
    )
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    eff = {"total_tokens": getattr(usage, "total_tokens", 0) or 0,
           "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0}
    return completion.choices[0].message.parsed, eff


def comprehend(model_payload: dict, gold: Comprehension, consumer_model: str, client=None):
    """Hand the (single) model to the consumer; return its answers + effort. `model_payload` is the
    arm-specific view (bare vs anchored) built by the driver."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    payload = {
        "model": model_payload,
        "meaning_questions": [{"qid": q.id, "concept_id": q.concept_id, "question": q.question}
                              for q in gold.meaning_questions],
        "use_task": {"goal": gold.use_task.goal},
    }
    t0 = time.time()
    answer, eff = _parse(client, consumer_model, _CONSUMER_SYSTEM, payload, ConsumerAnswer)
    eff["latency_s"] = round(time.time() - t0, 2)
    return answer, eff


def judge_meaning(gold: Comprehension, answer: ConsumerAnswer, judge_model: str, client=None):
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    by_qid = {a.qid: a.answer for a in answer.meaning}
    payload = {"items": [
        {"qid": q.id, "concept_id": q.concept_id, "answer_key": q.answer_key,
         "must_not_confuse_with": q.must_not_confuse_with,
         "consumer_answer": by_qid.get(q.id, "")}
        for q in gold.meaning_questions]}
    result, _eff = _parse(client, judge_model, _JUDGE_SYSTEM, payload, JudgeResult)
    return result.verdicts


# ---- scoring (pure python; offline-testable) ---------------------------------------------
def score_comprehension(gold: Comprehension, answer: ConsumerAnswer, verdicts: list[Verdict]) -> dict:
    v = {x.qid: x.label for x in verdicts}
    n = max(1, len(gold.meaning_questions))
    faithful = sum(1 for q in gold.meaning_questions if v.get(q.id) == "faithful")
    invented = sum(1 for q in gold.meaning_questions if v.get(q.id) == "invented")

    rel = set(gold.use_task.relevant_ids)
    got = set(answer.use_relevant)
    use_recall = len(rel & got) / max(1, len(rel))
    use_precision = len(rel & got) / max(1, len(got)) if got else 0.0

    amb = set(gold.ambiguous_ids)
    flagged = set(answer.flagged_ambiguous)
    ambiguity_flag_recall = len(amb & flagged) / len(amb) if amb else float("nan")
    # over_claim: of the gold-ambiguous concepts the consumer was actually asked about, how many it
    # answered WITHOUT flagging -- i.e. implicitly claimed to understand something it should have flagged.
    # This is the portability hazard whether the asserted meaning is invented (mid-tier confabulation) or
    # accidentally right; not flagging under-anchored content is the failure. Distinct from
    # confabulation_rate (wrong meaning, all Qs) and ambiguity_flag_recall (flag recall over all ambiguous).
    questioned = {q.concept_id for q in gold.meaning_questions}
    amb_q = amb & questioned
    over_claim = len(amb_q - flagged) / len(amb_q) if amb_q else float("nan")

    return {
        "meaning_score": round(faithful / n, 3),
        "confabulation_rate": round(invented / n, 3),
        "use_recall": round(use_recall, 3),
        "use_precision": round(use_precision, 3),
        "ambiguity_flag_recall": (round(ambiguity_flag_recall, 3)
                                  if ambiguity_flag_recall == ambiguity_flag_recall else ""),
        "over_claim": round(over_claim, 3) if over_claim == over_claim else "",
    }
