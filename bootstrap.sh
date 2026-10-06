#!/usr/bin/env bash
# Bootstrap the study's Python environment — idempotent, safe to re-run.
#
#   bash bootstrap.sh           create .venv (if missing) inside study/ and install deps
#   bash bootstrap.sh --force   delete .venv and rebuild from scratch
#
# Re-run this any time .venv goes missing (e.g. after refreshing the folder). It always
# operates inside study/ regardless of where you call it from, so the venv can never land
# in your home directory by mistake. It NEVER stores your API key: add OPENAI_API_KEY to
# study/.env yourself (this script only reminds you if it's absent).
set -euo pipefail

cd "$(dirname "$0")"                       # always operate in the study/ dir

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: '$PY' not found. Install Python 3.10+ (e.g. 'brew install python@3.12'), then re-run." >&2
  exit 1
fi
ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
major="${ver%.*}"; minor="${ver#*.}"
if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
  echo "error: Python $ver found, but 3.10+ is required." >&2
  echo "       Try:  PYTHON=python3.12 bash bootstrap.sh   (after 'brew install python@3.12')" >&2
  exit 1
fi

if [ "${1:-}" = "--force" ]; then rm -rf .venv; fi
if [ ! -d .venv ]; then
  echo "creating .venv with $PY ($ver) ..."
  "$PY" -m venv .venv
fi

set +u                                     # activate script references unbound vars under set -u
# shellcheck disable=SC1091
source .venv/bin/activate
set -u

echo "installing dependencies (openai + pydantic + pytest) ..."
python -m pip install -U pip >/dev/null
python -m pip install -e ".[openai,dev]"

echo "running offline sanity test ..."
python -m pytest -q tests/test_construct_cost_study.py || \
  echo "  (offline test reported an issue — environment is installed, but check the output above)"

echo
echo "environment ready.  venv: $(pwd)/.venv"
if { [ -f .env ] && grep -q OPENAI_API_KEY .env; } 2>/dev/null; then
  echo "OPENAI_API_KEY: found in study/.env"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY: found in your shell environment"
else
  echo "NOTE: no API key yet. The agent studies need it. Add it with:"
  echo "      printf 'OPENAI_API_KEY=%s\\n' 'sk-...your key...' > .env"
fi
echo
echo "from now on, in a new terminal:   cd <...>/study && source .venv/bin/activate"
