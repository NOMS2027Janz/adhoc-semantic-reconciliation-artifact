# Ad Hoc Semantic Reconciliation: Benchmark, Harness, and Validated Keys

This repository is the released artifact for the work on *ad hoc semantic reconciliation*: the
ability of cognitive agents to reconcile divergent network- and service-management models for the
occasion, with no model agreed in advance. It contains everything needed to reproduce the study and
its results, the benchmark and its validated keys, a non-cognitive baseline, and the instrumented
cognition traces, together with the code that produces them.

Archived at Zenodo: **DOI [10.5281/zenodo.23187404](https://doi.org/10.5281/zenodo.23187404)**.

## Companion papers

This artifact is the reference cited by two companion works:

- **IETF NMRG Internet-Draft**, *Ad Hoc Semantic Reconciliation* (the conceptual account and the
  four-scenario evidence).
- **NOMS paper**, *Portable Semantic Models and Ad Hoc Reconciliation by Cognitive Agents* (the
  measurement of portability and the empirical characterisation of reconciliation).

Both cite this archive for the benchmark, the harness, and the validated keys.

## The cognition note

**[`cognition/Agent_Cognition_in_Reconciliation.pdf`](cognition/Agent_Cognition_in_Reconciliation.pdf)** — *Agent cognition in ad hoc semantic reconciliation* — is a standalone note that reads the agents' reasoning turn by turn, through the two acts of the task. In the first, each side *lifts* its own data model into a semantic model, reading each concept's meaning from its structure. In the second, two agents *reconcile*: they establish which concepts denote the same thing, which are false cognates, and which have no counterpart at all, under two disclosure conditions (catalogue only, and explicit standard pointers). It shows, in the agents' own words and scored against the validated key, the cognitive moves the task calls for: reading meaning from structure, spending a decisive test where reasoning cannot settle a question, refusing a plausible false cognate, and declaring an honest gap. The agents run as a ReAct-style reasoning-and-acting loop, instrumented so the reasoning is recorded. The code that produces these traces, and the traces themselves, are in `cognition/` and `results/` (see below).

## What is here

```
src/reconcile/      the package: the semantic model and lift, the schema oracle (decisive
                    virtual experiment), reconciliation stacks (two-agent, baseline, classical,
                    reference matcher), metrics, and supporting modules
pipeline/           the run scripts: one per study, plus the figure generators
benchmark/cases/    the benchmark: independently built cases, each with two models and a key a
                    script derives and consistency-checks before any run
results/            the released run outputs (CSV summaries, transcripts, agent-lifted models)
figures/            the figures generated from those results
tests/              offline tests (no network; use an injected client)
cognition/          the orchestration materials behind the cognition note (see below)
demos/              interactive, self-contained HTML demonstrations of the scenarios
pyproject.toml      package and dependency definitions
bootstrap.sh        one-command environment setup
```

## Reproducing

Requirements: Python 3.10 or later, and (for the cognitive-agent runs) access to a language-model
API through an OpenAI-compatible endpoint.

```bash
# 1. environment (creates .venv and installs deps: openai -> pydantic, matplotlib, numpy, pytest)
bash bootstrap.sh
source .venv/bin/activate
#    equivalently:  pip install -e ".[openai,dev]"

# 2. provide your API key (never committed)
echo 'OPENAI_API_KEY=sk-...' > .env

# 3. run the offline tests (no API needed)
pytest -q

# 4. reproduce a scenario (one command per case) and regenerate its figures
python pipeline/two_agent_study.py --case config_tapi_teas
python pipeline/figures.py
```

Each reported cell is a mean over independent trials under the provider's default sampling, with no
seed pinned, so exact numbers will vary slightly run to run; the signature results reproduce. The
released `results/` let the reported numbers be checked against the actual run outputs without
spending any API.

## The cognition materials (`cognition/`)

The note described above (*The cognition note*) sits here, alongside the code that produces its
traces. `cognition/orch/` holds the instrumented and disclosure stacks, layered over the base
two-agent stack in `src/`:

```bash
# instrumented reconciliation (planning channel on)
python cognition/orch/two_agent_instrumented_transcript.py --case config_tapi_teas --experiment
# with the opening provenance-disclosure handshake
python cognition/orch/two_agent_disclosure_transcript.py --case config_tapi_teas --experiment
# the multi-trial comparison (catalogue vs disclosure)
python cognition/orch/run_trials_tapi_exp.py --trials 3
# the per-concept lift traces
python pipeline/lift_study.py --lift-only --trace --ref 0 --cases config_tapi_teas,config_cross_domain
```

The released traces these commands produce are under `results/` (`transcript_instrumented_*`,
`transcript_disclosure_*`, `trials/`, and `agent_lifted_models/`).

## Citing

Please cite the archived DOI above. See `CITATION.cff` for the full citation and the companion
papers.

## License

See `LICENSE`.
