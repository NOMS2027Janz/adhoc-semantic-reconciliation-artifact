"""Structured-output compatibility shim.

The drivers use `client.chat.completions.parse(..., response_format=PydanticModel)`, which sends a
strict `json_schema` response_format. OpenAI and Ollama honour it; some OpenAI-compatible providers
(e.g. DeepSeek) support only JSON *mode* (`response_format={"type":"json_object"}`) and reject the
schema form with a 400 "response_format type is unavailable".

`parse_compat` tries the strict parse first and, only on that specific incompatibility, falls back to
JSON mode: it injects the model's JSON schema into the prompt, asks for a bare JSON object, strips any
markdown fences, and validates into the same pydantic model. It returns the SAME completion object the
callers already use, with `.choices[0].message.parsed` populated, so no call site needs to change how
it reads the result or the usage/effort.
"""
from __future__ import annotations

import json


def _is_response_format_error(e: Exception) -> bool:
    s = str(e).lower()
    return "response_format" in s or "json_schema" in s or "response format" in s


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may be fenced or have prose around it."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    # fall back to the outermost {...} span
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            t = t[i:j + 1]
    return t


def parse_compat(client, model, messages, response_format):
    """Like client.chat.completions.parse, but falls back to JSON mode where json_schema is unsupported.

    Returns a completion whose choices[0].message.parsed is an instance of `response_format`."""
    try:
        return client.chat.completions.parse(model=model, messages=messages,
                                             response_format=response_format)
    except Exception as e:  # noqa: BLE001
        if not _is_response_format_error(e):
            raise
        schema = json.dumps(response_format.model_json_schema())
        aug = [dict(m) for m in messages]
        aug[-1]["content"] = (aug[-1]["content"]
                              + "\n\nReturn ONLY a single JSON object (no prose, no markdown fences) "
                                "that conforms to this JSON schema:\n" + schema)
        comp = client.chat.completions.create(model=model, messages=aug,
                                              response_format={"type": "json_object"})
        content = comp.choices[0].message.content
        comp.choices[0].message.parsed = response_format.model_validate_json(_extract_json(content))
        return comp
