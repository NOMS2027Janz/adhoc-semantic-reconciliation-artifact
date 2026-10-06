# Benchmark schema

Each case lives in `benchmark/cases/<case>/` and holds four plain-JSON files. The
formats are deliberately simple, so a case can be read, checked, and extended by
hand, without the harness.

## `model_a.json`, `model_b.json` — the two lifted semantic models

A lifted semantic model is a set of concepts. Each concept is one entry in an
ontology, named by a lexicon, with a gloss and a canonical example that
disambiguate it.

| field | meaning |
|-------|---------|
| `system`, `dialect`, `modules` | provenance: whose model this is and what it was lifted from |
| `concepts[]` | the lifted concepts |
| `concepts[].id` | stable id, unique within the model |
| `concepts[].label` | the lexical label (the surface term) |
| `concepts[].synonyms[]` | alternative surface terms |
| `concepts[].kind` | a shallow class (node, termination, service, …) |
| `concepts[].gloss` | a disambiguating definition (self-explanation) |
| `concepts[].example` | one canonical example (self-explanation) |
| `concepts[].ref` | the reference entry this concept was bound to, or `null` for a native gap |
| `concepts[].relations[]` | structural edges `{rel, target}` to other concept ids in the same model |
| `concepts[].instances[]` | concrete data the concept is realised by |

`ref` is the author's declared binding to the shared reference. A reference-aware
stack consumes it; a reference-blind stack must ignore it and infer
correspondences from labels, synonyms, kinds, and glosses alone.

Instance-related information plays three distinct roles in the study, carried by
different structures; a reader should not assume a concept's `instances[]` list
flows through all three. First, the `concepts[].instances[]` here are **grounding
evidence**: concrete realisations the lifting and reconciliation agents read to
expose naming conventions and structural usage, especially where a label could
mislead. Second, the **translation and verification proposals** exercised in the
round-trip and invariant checks are separate constructs, generated during those
checks rather than drawn from this list. Third, **individual co-reference records**
— deciding which individual on one side is which on the other — are maintained as a
changing correspondence map of their own. The three are related but not the same
field, and only the first is the `instances[]` defined above.

## `reference.json` — the thin shared reference

An identity-only anchor per concept: `id`, preferred `label`, `synonyms`, a
shallow `class`, a disambiguating `definition`, and one canonical `example`. It is
small on purpose and is not a model of the domain.

## `gold.json` — the gold-standard reconciliation

The correct answer the harness scores against.

| field | meaning |
|-------|---------|
| `correspondences[]` | the correct `{a, b, ref}` cross-model concept pairs |
| `false_cognates[]` | `{a, b, why}` pairs that must NOT be proposed (silent-error traps) |
| `residual.a_only[]`, `residual.b_only[]` | native gaps, one side only, and how each is closed |
| `residual_by_placement` | items expected to remain open at each cognition placement |
| `invariants[]` | what a correct translation must preserve |
| `verification_by_placement` | the verification method each placement admits |

## Cognition placement

The placement names how many of the two systems are live, interrogable agents:
`both_cognitive`, `one_inert`, `both_inert`. It selects the confirmation regime,
the residual expected, and the verification method admitted.

An **inert** side is not absent — it is still fully present, keeping its structure
(`relations`) and its data (`instances`); it simply cannot explain itself in words.
So under `one_inert` the agent stack shows the inert side its label, kind,
relations, and instances, but withholds its gloss, synonyms, example, and reference
binding. The live agent must reconstruct the inert side's meaning from that
structure and data, which makes inertness *add* reconstruction work rather than
remove input. Under `both_cognitive` both sides volunteer their gloss and example
as usual.
