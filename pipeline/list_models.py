#!/usr/bin/env python3
"""Print the OpenAI model IDs this account can access (reads study/.env for the key).

  python pipeline/list_models.py           # all ids, sorted
  python pipeline/list_models.py gpt-5     # only ids containing 'gpt-5'
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from run import load_dotenv   # noqa: E402


def main() -> int:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; see notes/SETUP_OPENAI.md", file=sys.stderr)
        return 2
    from openai import OpenAI
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    ids = sorted(m.id for m in OpenAI().models.list())
    for i in ids:
        if needle in i:
            print(i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
