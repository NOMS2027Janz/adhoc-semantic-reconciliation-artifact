#!/usr/bin/env python3
"""Sanity-check a LOCAL (or any OpenAI-compatible) model endpoint for the reconciliation harness.

The harness reaches every model through the OpenAI Python client, which honours the
OPENAI_BASE_URL environment variable. To use a local model (e.g. served by Ollama at
http://localhost:11434/v1), set:

    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=ollama            # any non-empty string; local servers ignore it

and then run the normal drivers with --model <local-model-name>. This script confirms, before you
spend a whole experiment, that the endpoint is reachable AND that it can produce the STRUCTURED
output the drivers rely on (chat.completions.parse with a JSON schema). If both checks pass, the
chat-completions-based experiments (E1, E5, E6, E9, E9-conflict, E7/E7b, E8, E12, E13) will run
against this model.

    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
        python pipeline/check_local_model.py --model qwen2.5:32b-instruct
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel


class _Correspondence(BaseModel):
    a_id: str
    b_id: str
    same: bool
    reason: str


class _Probe(BaseModel):
    corresponds: list[_Correspondence]


PROMPT = (
    "You reconcile two tiny semantic models of one network. Decide which concepts denote the same "
    "thing, by meaning not label. Model A: [{id:a1,label:'committed rate',gloss:'the customer IP "
    "service bandwidth commitment'}]. Model B: [{id:b1,label:'policer rate',gloss:'the ingress "
    "committed rate enforced for the service'},{id:b2,label:'line rate',gloss:'the physical port "
    "capacity'}]. Return one entry per A concept with the B concept it corresponds to (or same=false)."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local model name, e.g. qwen2.5:32b-instruct")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                    help="override OPENAI_BASE_URL (e.g. http://localhost:11434/v1)")
    args = ap.parse_args()

    base = args.base_url or os.environ.get("OPENAI_BASE_URL")
    print(f"endpoint : {base or '(default OpenAI)'}")
    print(f"model    : {args.model}")

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: `openai` not installed in this venv (pip install openai)", file=sys.stderr)
        return 2

    client = OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "local")) if base \
        else OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "local"))

    # 1) reachability + a plain completion
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nFAIL (reachability): {e}", file=sys.stderr)
        print("  - is the server running?  (ollama serve, or the app)\n"
              "  - is the model pulled?    (ollama pull <model>)\n"
              "  - is OPENAI_BASE_URL right? (…/v1)", file=sys.stderr)
        return 1
    print(f"\n[1/2] plain completion  OK  ({time.time()-t0:.1f}s)  -> "
          f"{(r.choices[0].message.content or '').strip()[:40]!r}")

    # 2) structured output (what the drivers actually need) -- via the same compat shim the drivers
    #    use, so a PASS means the drivers will work on this endpoint (strict json_schema OR the
    #    json-mode fallback for providers like DeepSeek).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from reconcile.compat import parse_compat  # noqa: E402
    t0 = time.time()
    try:
        comp = parse_compat(
            client, args.model,
            [{"role": "system", "content": "Return structured output only."},
             {"role": "user", "content": PROMPT}],
            _Probe,
        )
        parsed = comp.choices[0].message.parsed
    except Exception as e:  # noqa: BLE001
        print(f"[2/2] structured output  FAIL: {e}", file=sys.stderr)
        print("  Neither strict json_schema nor the json-mode fallback produced valid structured "
              "output on this endpoint/model. Try a model with stronger JSON adherence, or "
              "report the error.", file=sys.stderr)
        return 1

    if parsed is None:
        print("[2/2] structured output  FAIL: parsed came back empty", file=sys.stderr)
        return 1

    print(f"[2/2] structured output  OK  ({time.time()-t0:.1f}s)")
    for c in parsed.corresponds:
        print(f"        {c.a_id} ~ {c.b_id}  same={c.same}  ({c.reason[:50]})")
    # a light correctness hint (not scored): a1 should map to b1, not b2
    good = any(c.a_id == "a1" and c.b_id == "b1" and c.same for c in parsed.corresponds)
    print(f"\n  sanity: committed-rate→policer-rate found = {good} "
          f"(a weak model may miss it; that's fine — this only checks the plumbing)")
    print("\nPASS — this endpoint can drive the chat-completions experiments. Run e.g.:\n"
          f"  OPENAI_BASE_URL={base} OPENAI_API_KEY=local \\\n"
          f"    python pipeline/authority_conflict.py --model {args.model} --trials 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
