# The ad hoc semantic reconciliation benchmark

A small, open benchmark for **reconciling divergent network models with cognitive agents**. Each case
is two lifted semantic models (plus, where relevant, a thin shared reference and planted traps) and a
**gold standard derived from the models and validated**, so a reconciliation stack can be scored
against a fixed answer that cannot drift. The benchmark is designed to be read, checked and extended by
hand — every case is plain JSON — and to grow by contribution.

This document is the front door. The exact file formats are in [`schema.md`](schema.md); how to run a
reconciliation over a case is in the [study README](../README.md).

## What is in it

Ten cases today, spanning the four operational settings of the study and several sub-studies:

- **Setting 1 · configuration** — `config_tapi_teas`, `config_big_hard` (two standard models of one
  network), with sub-studies `instance_hard` (instance co-reference), `verify_hard` (verification), and
  `scaling_otn` (the N-vs-N² scaling count).
- **Setting 2 · intent** — `intent_hard` (refinement, negotiation, lifecycle, policy) and
  `intent_endpoints` (endpoint instance co-reference).
- **Setting 3 · cross-domain** — `config_cross_domain` (two private models, no public standard).
- **Setting 4 · observability** — `config_observability` (alarm↔anomaly, RFC 9940) and `obs_instance`
  (alarm instance co-reference).

`manifest.json` is the machine-readable index (case, family, setting, files, and counts), regenerated
from the cases by the tool below.

## Case families

A case belongs to one family, recognised by its signature files:

- **schema** — `model_a*`, `model_b*`, `reference.json`, `gold.json` (the core reconciliation task).
- **instance** — `individuals_a*`, `individuals_b*`, `instance_reference.json`, `instance_gold.json`
  (entity resolution over populated models).
- **intent** — `catalogue.json`, `intents.json`, `intent_reference.json`, `intent_gold.json`
  (refinement/negotiation rather than equivalence).
- **scaling** — several `model_*` plus `reference.json`, no gold (a structural count, not a scored run).
- **verify** — `proposals.json`, `verify_gold.json` (verification in isolation).

## What is scored

A reconciliation stack is run over a case at a point on the **cognition spectrum** (`both_cognitive`,
`one_inert`, `both_inert`) and scored against the gold on: **precision** (of what it proposes, the
fraction correct), **resolved fraction** (of the true correspondences, the fraction it commits, the
rest referred to the residual), **surviving false cognates** (planted traps taken), and the
**residual**. For a language-model stack, effort is also recorded (reasoning tokens, total tokens,
latency). A resolved fraction below one is honest deferral, not error — read it alongside precision.

## The packaging tool

`pack.py` validates cases and builds the manifest, entirely offline (no model access):

```bash
python benchmark/pack.py validate          # check every case is well-formed
python benchmark/pack.py validate <case>   # check one case
python benchmark/pack.py manifest          # (re)write manifest.json
```

`validate` checks that each case has the files its family requires, that every JSON parses, and — for
schema cases — that concept ids are unique, that structural relations resolve within their model, that
declared reference bindings resolve to the reference, and that gold correspondences and traps reference
real concept ids. It exits non-zero on any error, so it can gate CI.

## Contributing a case

The benchmark is meant to grow, and contributions are welcome — especially cases grounded in real
operational models rather than constructed ones.

1. Create `benchmark/cases/<your-case>/` and populate it following [`schema.md`](schema.md) for your
   family.
2. Run `python benchmark/pack.py validate <your-case>` until it reports no errors.
3. Run `python benchmark/pack.py manifest` to register it.
4. Open a pull request. A good case carries a real set of correspondences, at least one **false-cognate
   trap** (a look-alike that must be refused), a **thin reference** the two sides can bind through, and
   a **gold derived from the models and validated**, not hand-written after the fact.

**For carrier and NMOP contributors:** a case need not expose sensitive data. A model can take part by
publishing only its **lexicon layer** — the labels, synonyms, kinds, glosses, examples and reference
bindings of its concepts — while its relationships and instance data stay private or are reduced to a
bounded, de-identified sample. That is the same lexicon-layer route the reference-lexicons draft
describes, and it is enough to build a scorable case. A real model pair, a real alarm/anomaly stream,
or a real instance-resolution sample from a production network would each be a valuable addition.

## Licensing

Choose a license before publishing the benchmark, and record the provenance and any redistribution
terms of each contributed case in its folder.
