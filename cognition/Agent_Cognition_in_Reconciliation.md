# Agent cognition in ad hoc semantic reconciliation

*The lift and the negotiation, instrumented turn by turn.*

## 1. What this note shows

This note is a standalone illustration of agent cognition at work in ad hoc semantic reconciliation. It follows the task through its two acts. In the first, each side *lifts* its own data model to a semantic model: reading its concepts from their structure and data and writing down what each one means. In the second, two agents, each holding one model privately, *reconcile*: they establish which of their concepts denote the same thing, which are false cognates, and which have no counterpart at all. Throughout, the agents run as a reasoning-and-acting loop in the ReAct style [react], reasoning about the state, taking an action (asking, proposing, testing), observing the result, and iterating; the only addition is that we have instrumented the reasoning so that it is recorded turn by turn and can be read rather than inferred.

The point is not a benchmark score. It is to show, in the agents' own words, the cognitive moves the task calls for: reading meaning from structure, forming and deferring hypotheses, spending a decisive test where reasoning cannot settle a question, refusing a plausible-looking false cognate, declaring an honest gap, and knowing when to stop. Every outcome is scored against a human-validated key, so when an agent says it is confident we can check whether it was right to be. The benchmark, the cases, the harness, and these instrumented runs are released in the accompanying archive.

Two models carry the illustration. The main one reconciles a connection-oriented transport catalogue written in ONF TAPI against a traffic-engineering catalogue written in IETF TEAS: two real, related standards, which makes the false cognates subtle. A second, harder case reconciles two bespoke dialects, a transport OSS against an IP/VPN controller, where several concepts have no counterpart at all and two share a label while meaning different things.

## 2. The first act: the lift

Reconciliation presupposes that each side has a *semantic* model to offer, not merely a data model. A data model is a schema with data in it: concepts named by labels, given a kind, wired to one another by structural relations, and realised by records. A complete, portable lifted semantic model is that same structure carried with the layer that makes it interpretable by another party. For each concept it holds the lexical surface (label and synonyms), the schema structure (kind and relations), the realising data (instances), a disambiguating gloss of what the concept means with one worked example, and the pragmatic and provenance binding: which system it belongs to, which published model or dialect it is written in, and, where one exists, the shared reference entry its author bound it to. The test of a complete lift is operational: another cognitive agent, reading it, can decide whether one of its own concepts denotes the same thing.

Producing that explanation layer is the lift, and how much the agent has to work from depends on how it is situated. A fully embedded agent, one that sits inside the operational system rather than being handed a static export, can always complete a fully portable lift, because it has more than the schema available: it can interrogate the live systems for configuration and state, read whatever documentation, standards, and existing glosses it can reach, examine as much realising data as it needs, and run virtual experiments, provisioning and operating a concept to observe what it actually does. Whatever the surface underdetermines, embedding gives it the means to resolve.

This note deliberately tests the floor beneath that. The lifting agent here is given one side's data-model surface only: each concept's label, its synonyms, a shallow kind, its structural relations to other concepts in the same model, and the concrete instances that realise it, together with the name of the system and dialect it is lifting. The glosses, the other model, and any shared reference are withheld, and the agent neither interrogates any live system nor runs any virtual experiment during the lift: it reasons once over the material in front of it and writes, for each concept, a one-sentence gloss and one worked example, grounding the reading in the kind, the relations, and the instances rather than restating the label. The virtual experiment enters later, in the reconciliation, not here. The question this poses is whether the lift is agent-performable from the surface alone, the hardest case and the one an embedded agent could always improve on with fuller access; the traces below show that on these models it is.

What the instrumentation shows is that the reading is genuinely inferential. Consider two concepts from the TAPI side. For the service-interface-point, the agent wrote:

> A service-interface-point is a node-associated access point through which a requester selects or reaches an offered connectivity service at a particular service or layer interface.

and recorded, as its evidence, that it read the kind "access", the on-node relation, the reciprocal service relations to the connectivity-service, and the two customer-facing instances, and that it ruled out the link-termination-point, which terminates topology links, and the connection-endpoint. For the connectivity-service it wrote:

> A connectivity-service is a requested end-to-end connectivity construct between service interface points, with specified layer characteristics, whose realization is represented by connection endpoints over a topology.

reading the kind "service", the relations to the service-interface-point, the connection-endpoint, and the topology, and the end-to-end ODU2 instance, and ruling out the topology link, which is an individual adjacency rather than an end-to-end intent.

The TEAS side is lifted independently and in ignorance of the TAPI side. There the agent reads the tunnel-termination-point as "a node-local access endpoint from which a TE tunnel is originated into the TE topology", from its kind "termination", its on-node and originates relations, and its tunnel-head instances, ruling out the tunnel connection endpoint; and the TE tunnel as "an engineered transport service originating at a tunnel termination point and carried across a TE topology", from its kind "service" and its relations, ruling out the TE link.

These independently produced glosses already set up the hard questions the reconciliation will face. The TAPI service-interface-point and the TEAS tunnel-termination-point are both node-local access endpoints, and their labels invite a match, yet one is an external client-service hand-off and the other an internal tunnel trail head. Nothing in the two surfaces settles that; it is exactly the kind of question the second act has to resolve.

## 3. The second act: the setup

The two agents hold their models privately and reconcile under real information asymmetry. Each is the sole authority on its own side. A correspondence is confirmed only when one side proposes it and the owning side ratifies it, or when both propose it independently: neither side alone decides a cross-model correspondence. When a question cannot be settled by talk, an agent may call a decisive virtual experiment: a schema-level oracle provisions the proposed pairing, operates it, and reports whether the invariants a correct translation must preserve actually hold. The experiment budget is finite, so the agents must spend it deliberately. The loop is bounded, and each turn an agent also fills a planning channel, in its own first person: its reading of where things stand, the one thing it most needs to settle next and why, its strategy for the turn, its estimate in the unit interval of how close it is to a confident close, and a decision to continue, stop, or escalate. That channel is an observational window, not a new instruction to perform well.

A word on how the agents begin. The two agents start already in contact. Before the first turn shown, each has advertised its catalogue to the other: for every concept, an id, a label, and a kind (the coarse class, such as access, service, or node). That is all either side has of the other at the outset; the glosses, examples, relations, and instances that give each concept its meaning stay private, and are drawn out only through the questions the agents ask. In a deployment the two sides would first have to find each other, open a channel, and exchange these catalogues; we take that step as already done, so the trace opens with each agent holding the other's surface catalogue and nothing more. That catalogue is a standing input to each agent, not a move it made, which is why it is not itself an entry in the transcript.

What the agents do *not* begin with, in the first case below, is any statement of which published model each side speaks. They have the other's labels and kinds; they have not been told that one side is ONF TAPI and the other IETF TEAS. The second case adds exactly that, and the difference it makes is the subject of Section 5.

## 4. Case 1: reconciliation from surfaces alone

In the first case the agents reconcile from the exchanged catalogues and whatever they can draw out of each other, with no standard named. The reasoning-and-acting loop is visible from the first turn. Agent A opens by declining to guess:

> This is the first reconciliation turn. My model has ten concepts, and B advertises ten similarly typed concepts, suggesting several plausible one-to-one correspondences. None is settled, however, and labels alone cannot establish semantic identity, especially for service versus tunnel, adaptation link versus supporting TP, layer qualifier versus OTN label, and CEP versus tunnel-specific endpoint.

Its stated need is "semantic descriptions of B's concepts, because proposing solely from the catalogue would risk false cognates", and its strategy is to ask about all ten and defer every proposal and experiment until meanings arrive. This is hypothesis formation held in check: the surface is enough to see where the traps probably are, not enough to commit.

Once meanings have been exchanged, A separates what talk has settled from what only a test can settle. It proposes the three correspondences it now holds with confidence (termination point, transitional link, connectivity matrix), and it reserves the decisive experiment for the pairs where surface similarity is seductive but unproven:

> I most need decisive results for the service/tunnel and endpoint-related candidates, because their examples appear operationally similar even though the models assign them to different kinds of parent constructs; a surface comparison cannot establish preserved identity.

It tests connectivity-service against TE tunnel, layer-qualifier against OTN label, and two more; all four come back confirmed. The judgement of what to test, and what to spend the budget on, is the orchestration move: it does not test the three it is already sure of, and it does not test the service-interface-point, which it has reasoned has no counterpart at all.

The next turn shows the loop revising its plan on the evidence. Agent B, seeing the topology identity confirmed, reopens two concepts it had been ready to set aside:

> Although A initially marked node and link unmatched because of per-layer versus unified modeling, the confirmed topology identity makes those candidate translations important to test. The confirmed service/tunnel identity likewise makes the otherwise semantically distinct TTP/SIP endpoint pairing worth testing rather than dismissing by terminology.

It tests node and link, which confirm, and it asks for one more test, of the tunnel-termination-point against the service-interface-point, the one genuine trap. That test cannot run: the experiment budget is spent. Here the loop meets its limit, and the cognition on display is the one that matters most for reliability. Rather than guess, with both the labels and the confirmed-identical parents pulling toward a match, the agents fall back to the authoritative glosses and refuse:

> Although the t.sip/i.ttp experiment could not run because of the experiment budget, the authoritative glosses distinguish them: t.sip is an external client-service hand-off, whereas i.ttp is an internal tunnel trail head. Thus t.sip has no counterpart in B's catalogue.

Both sides mark their endpoint concept unmatched and close at full confidence. The run reaches nine of nine true correspondences, with precision intact, the single trap declined, and the budget-exhausted pair refused with its reason on the record.

That is the repertoire the task calls for, shown on one good run. But this case is not uniformly that good, and the honest picture needs the distribution. Across three trials the catalogue-only arm scored recall 0.78, 0.89, and 1.00, for a mean of 0.89, at precision 1.00 throughout. The variation is not random noise in the score; it is a specific cognitive lapse. In the weaker runs the agents dismiss a true pair by reasoning rather than testing it. In one trial, for instance, Agent A marked the connectivity-service and the layer-qualifier as having no counterpart without ever running the experiment on them, and both are real matches; the run closed at full confidence having missed them. The decisive instrument was available and was not spent on the pairs that most needed it. This is the failure the second case bears on.

## 5. Case 2: reconciliation with explicit standard pointers

The second case changes one thing. It makes the opening exchange explicit and adds to it: each side also advertises the published model it speaks, its system and dialect, here ONF TAPI on one side and IETF TEAS on the other. The handshake is now a recorded event at the head of the trace, and each agent knows not just the other's labels but which standard produced them. Everything else is identical: the same case, the same tool, the same budget.

The effect is not that the agents coast on the standard names. It is that knowing the provenance sharpens their judgement of which pairs are worth the decisive test. In a representative run, Agent A, once meanings are exchanged, proposes the structural correspondences it is sure of and in a single turn runs the experiment on every consequential pair at once, node scope, adaptation representation, service versus tunnel, qualifier versus resource label, multi-layer versus per-layer topology, and the connection endpoint:

> I test node scope, adaptation representation, service versus tunnel, qualifier versus resource label, multi-layer versus per-layer topology, and connection endpoint semantics.

All six confirm; Agent B ratifies; the service-interface-point and the tunnel-termination-point are correctly left unmatched. The run reaches nine of nine. Across three trials the standard-pointer arm scored recall 1.00, 1.00, and 1.00, at precision 1.00, spending five or six experiments per run against the catalogue arm's two to four.

The comparison is the finding, and it runs opposite to the natural worry that naming the standards would let the agents reason their way past the evidence. It is the catalogue-only arm, lacking the provenance, that more often dismisses a true pair without testing it; the standard-pointer arm identifies the consequential pairs more reliably and spends the decisive test on them, reaching full recall every time at the same perfect precision.

| Condition | Recall (3 trials) | Mean recall | Precision | Experiments |
|---|---|---|---|---|
| Case 1: catalogue only | 0.78, 0.89, 1.00 | 0.89 | 1.00 | 2 to 4 |
| Case 2: standard pointers | 1.00, 1.00, 1.00 | 1.00 | 1.00 | 5 to 6 |

Case 1 reconciles from surfaces alone. Case 2 reconciles when each party also tells the other which published model it speaks, and does better for a specific, legitimate reason: the disclosed provenance does not replace the decisive experiment, it tells the agents where to point it.

## 6. A harder seam: bespoke dialects, same labels, and real gaps

The third case stresses two things the TAPI/TEAS pair shows only mildly: concepts that share a label while meaning different things, and concepts that genuinely have no counterpart at all. It reconciles a bespoke transport OSS against a bespoke IP/VPN controller, so the disclosed dialects are each side's own vocabulary rather than a shared published standard, and the standard-pointer disclosure buys less.

The agents handle the genuine gaps cleanly. The IP/VPN side correctly marks its VLAN identifier and its IP-service object as having no counterpart in the transport model, recognising that the right answer for a concept can be that nothing in the other world corresponds to it. And they correctly refuse the one planted false cognate: both sides carry a concept labelled "grade", and the agents decline to match them, reading the two as different despite the shared word.

But this run also shows the Section 4 failure mode persisting, and shows why it is the central risk. Both sides also carry a concept labelled "latency", and here the shared label *is* a true correspondence. Telling the true same-label pair (latency) from the false one (grade) is exactly the question the decisive test exists to answer. The agent did not test latency; it reasoned the pair away as it did grade, and marked both unmatched, so it refused a true correspondence along with the false one. It likewise dismissed the hand-off against the attachment, another true pair, without a test. The run ends having refused the trap and found the gaps, but short two real correspondences it declined to verify, on a seam where the bespoke provenance gave it less to go on.

The lesson is the same one the trials in Section 5 make quantitatively. The decisive moment in reconciliation is the choice between settling a plausible pair by reasoning and settling it by test. Getting that choice right is what separates a confident wrong answer from a correct one, and knowledge of provenance helps most precisely when it improves that choice.

## 7. What the traces show

Read together, the runs exhibit a consistent cognitive repertoire. The agents read meaning from structure rather than from labels, both in the lift and in the negotiation. They form hypotheses from the surface and hold them until they have grounds. They separate what dialogue has settled from what only a decisive test can settle, and they spend a finite budget on the pairs that matter. They revise a plan when an intermediate result changes the picture. They refuse a plausible false cognate, with a stated reason, even when the labels and the confirmed parents pull toward a match. They declare an honest gap when a concept has no counterpart. And they close on a reading of their own state rather than on a fixed number of steps. This is the ReAct pattern [react], reasoning interleaved with tool-acting and observation, made observable by instrumenting the reasoning channel, and extended with a decisive-experiment tool and an explicit stop-or-continue judgement.

The traces also locate the single largest risk, and it is a cognitive one, not a tooling one. The reliability of the whole depends on the agent choosing to test a plausible pair rather than reason it away, exactly when surface similarity and prior knowledge make reasoning feel sufficient. The catalogue-only runs fail here more often; the harder bespoke case fails here too. The one condition that reduced it to zero across three trials was the one that gave the agents better grounds for choosing what to test: an explicit, named provenance. The lesson carries directly to any orchestrating agent that holds a decisive instrument and a store of prior knowledge. Prior knowledge should tell such an agent where to point the instrument; it must not become a licence to skip it. "I already know what this is" is precisely the state in which the decisive test is most worth spending.

A note on evidence. The lift examples, the Case 1 walkthrough, and the Case 2 walkthrough are each a single representative run, chosen to show the cognition clearly; the Section 5 comparison and its recall figures are over three trials per arm, and precision and the refusal of false cognates were stable across every run. The cross-domain case is a single run and is presented as an illustration of the same-label and true-gap behaviours and of the persisting failure mode, not as a measured comparison. All runs, their transcripts, and the scoring are in the accompanying archive.

## References

[react] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," International Conference on Learning Representations (ICLR), 2023. arXiv:2210.03629.
