# Research Narrative And Method

This document gives the end-to-end narrative arc and methodological arc for `LLM_Survey`: what intellectual problem the repo is solving, why the implementation choices look the way they do, what evidence each layer is meant to produce, and how the repository should be read as a piece of educational research infrastructure rather than only a software project.

## Executive Frame

The core research question behind this repository is simple but nontrivial:

How much of the contemporary large-language-model story can be reconstructed as a local, inspectable, NumPy-first implementation program without collapsing into either empty survey summary or misleading pseudo-reproduction?

The survey paper provides the thematic map. This repository provides the runnable counterpoint. The research contribution is therefore not a new model family. It is a reproducible translation layer from survey-scale conceptual breadth into mechanism-level local artifacts, provenance-aware analytical tables, and comparable JSON outputs.

The repo is built on five commitments:

1. coverage must be explicit
2. mechanisms must be runnable
3. provenance must be inspectable
4. comparisons must be normalized enough to be useful
5. limitations must remain visible rather than hidden by polished packaging

## Narrative Arc

### Act I: From Survey Breadth To Local Mechanism

The paper spans architectures, objectives, data, alignment, prompting, retrieval, evaluation, systems, and applications. A normal survey reader sees breadth but not executable continuity. The first narrative move in this repo is to turn breadth into a dependency spine:

- recurrent models and seq2seq before transformers
- attention before retrieval and RAG
- pretraining objectives before adaptation
- utilization behaviors before risk evaluation
- systems constraints before publication-level comparison

This matters because LLM capability is not one idea. It is the cumulative effect of architectural choices, training objectives, data allocation, adaptation layers, retrieval augmentation, and deployment-time control. A faithful educational reconstruction has to preserve that cumulative logic.

> **Story Box 1: The repo is not asking “what is an LLM?”**
>
> It is asking “what chain of design decisions makes the modern LLM landscape legible enough to rebuild in miniature?”

### Act II: From Miniature Mechanisms To A Survey-Scope Program

The second narrative move is to reject the false choice between trivial toy demos and impossible full reproductions. The repository uses “mechanism-level lite implementations” as a middle tier:

- small enough to run locally
- structured enough to preserve causal ideas
- instrumented enough to emit reports
- scoped enough to remain honest about what is omitted

This is the central methodological compromise of the project. The repo does not reproduce industrial training runs, web-scale corpora, or benchmark-scale evaluations. Instead, it isolates one claim at a time:

- what causal masking changes
- what prefix visibility changes
- what retrieval changes
- what position changes
- what adaptation changes
- what memory or batching changes
- what red-teaming or truthfulness probes expose

> **Method Box 1: Lite, But Not Arbitrary**
>
> A module counts as acceptable only if it preserves the mechanism under discussion, produces a canonical artifact, and fits into a shared reporting layer. “Toy” is allowed. “Unprincipled” is not.

### Act III: From Individual Demos To A Comparative Research Surface

A repo full of isolated demos is still not a research surface. The third narrative move is to standardize outputs:

- canonical JSON reports
- benchmark family normalization
- provenance tables
- module matrices
- figure panels
- publication docs

This converts local experiments into a comparative object. The question changes from “does this demo run?” to:

- what family does it belong to?
- what is the evidence file?
- is it donor-derived or repo-authored?
- what metrics are comparable to what?
- where is the mechanism faithful, and where is it only survey-mapped?

> **Story Box 2: The repo becomes readable at three resolutions**
>
> A newcomer can read the README.
> A technical reader can inspect the module and its demo JSON.
> A research-minded reader can follow the provenance and benchmark layers across modules.

### Act IV: From Coverage To Fidelity

Coverage completion was necessary but insufficient. Once the repo reached tracked-scope completeness, the problem changed. The highest-value work was no longer “add another row.” It became:

- replace generator-backed placeholders with dedicated modules
- upgrade weak demos into mechanism-level versions
- make provenance sharper
- turn dashboards into evidence-backed tables
- align figures and tables with publication logic

This is the fidelity-hardening phase. The repo’s later evolution is therefore best understood as an argument about epistemic quality:

- completion says the topic is present
- provenance says where it came from
- fidelity says how strongly it represents the underlying research idea

> **Method Box 2: Coverage Is Binary, Fidelity Is Layered**
>
> Coverage answers whether a topic exists in the repo.
> Fidelity answers how faithfully the repo captures the mechanism, evidence, and comparison logic of that topic.

### Act V: From Codebase To Research Instrument

The final narrative move is to package the repo so it can be read as a research instrument:

- publication-facing docs first
- package-facing install surface for reusable code
- confidence summaries visible before matrices
- figures paired with callout boards
- stale artifacts reconciled
- provenance separated into mechanism, survey-map, and resource/reporting layers
- versioned release packaging for reproducible snapshots

At this point the project stops looking like a pile of demos and starts looking like an educational observatory of the survey.

> **Story Box 3: The intended user is not only a coder**
>
> The intended user is also a reader who wants to understand how LLM ideas connect, where the evidence comes from, and which parts of the repo are stronger or weaker representations of the literature.
>
> That is why the repo now has two surfaces: a publication/research surface in the repository and a smaller package surface for reusable imports. The tag and release boundary make that distinction legible.

## Methodological Arc

## Stage 1: Scope Construction

The repository began by defining an explicit paper-scope target list rather than letting scope sprawl emerge from ad hoc coding. This produced a tractable research program:

- enumerate survey themes
- split them into modules
- mark what is already donor-covered
- identify what must be repo-authored
- track completion numerically

The purpose of explicit scoping was not only project management. It was methodological discipline. A survey-inspired implementation repo becomes misleading very quickly if it quietly drops difficult topics such as contamination, truthfulness, privacy, preference optimization, or long-context behavior.

> **Method Box 3: Scope Before Implementation**
>
> The paper is treated as a target surface. Implementation starts only after the target surface is named, categorized, and tracked.

## Stage 2: Reuse-First Extraction

The second methodological move was reuse-first extraction from the local donor repo `pageman/sutskever-30-implementations`.

That reuse policy had strict constraints:

- reuse ideas and mechanisms, not wholesale notebook copying
- normalize notebook logic into importable modules
- keep provenance explicit
- prefer donor-linked foundations where they strengthen mechanistic fidelity

This approach was especially valuable for:

- RNN and LSTM language models
- attention and transformer basics
- scaling-law intuition
- retrieval and RAG
- long-context position effects
- multi-token prediction
- pipeline-style systems intuition

> **Story Box 4: Reuse Was A Fidelity Strategy**
>
> The donor repo was not just a convenience. It was the main way to keep the local educational implementations anchored to existing mechanistic explanations rather than inventing arbitrary substitutes.

## Stage 3: Core Primitive Consolidation

Before broad module expansion, the repo extracted a shared core:

- attention primitives
- recurrent primitives
- transformer primitives
- data helpers
- metrics
- reporting schema

This stage is methodologically important because it reduced “topic coverage” from a set of isolated notebooks into a system with reusable internal language. Once the core existed, later modules became comparable by construction rather than only by post hoc documentation.

## Stage 4: Dedicated Module Expansion

The repo then expanded through waves:

- foundational models
- pretraining and data modules
- retrieval and utilization modules
- evaluation and risk modules
- adaptation/alignment modules
- architecture and systems modules
- applications and stretch modules

Each wave followed the same method:

1. implement a narrow mechanism
2. add a runnable local demo
3. emit a canonical report
4. add smoke coverage
5. integrate the artifact into the publication layer

> **Method Box 4: Every Module Must Terminate In Evidence**
>
> The endpoint is not code alone. The endpoint is a reproducible artifact that can be indexed, compared, and cited inside the repo.

## Stage 5: Generator Elimination

At one point, broad paper-scope completion included a generic paper-scope generator. That solved the coverage problem but created a fidelity ceiling. The next methodological phase was therefore elimination of generic fallback coverage:

- identify the weakest generator-backed topics
- replace them with dedicated modules
- push the generic bucket to zero

This was a major epistemic improvement. A “complete” repo with generic placeholders is not equivalent to a complete repo with dedicated mechanisms.

> **Story Box 5: Zero Generic Coverage Became A Research Quality Threshold**
>
> The point of driving generic coverage to zero was not aesthetics. It was to ensure that every tracked topic had a concrete technical home.

## Stage 6: Provenance Hardening

Once all tracked topics had dedicated homes, the next methodological issue was trust. Provenance hardening answered:

- which modules are donor-derived?
- which are repo-authored?
- which are computed summaries?
- which evidence file supports each row?
- which supporting artifact is canonical?

This created a distinction between three different kinds of truth claims inside the repo:

- mechanistic truth claims from runnable modules
- structural truth claims from survey-map tables
- organizational truth claims from reporting and resource layers

> **Method Box 5: Provenance Is A First-Class Research Object**
>
> Provenance is not metadata bolted onto the end. It is part of the argument that the repo is making about its own reliability.

## Stage 7: Benchmark Normalization

Comparability is dangerous if done loosely. Different demos report different raw metrics, scales, and failure modes. The repo therefore evolved from flat aggregation to family-local comparison:

- pretraining-data families
- training-efficiency families
- systems-efficiency families
- utilization-retrieval families
- utilization-reasoning families
- evaluation-context families
- evaluation-risk families
- adaptation families

Then those were tightened further into narrower families and family groups. The methodological benefit is modest but real: metrics are less likely to be compared outside their proper context.

> **Story Box 6: Comparability Needed Governance**
>
> Without family-local normalization, the benchmark layer would have produced attractive but shallow rankings. The family structure makes comparison more defensible.

## Stage 8: Publication Packaging

The last major methodological stage was publication packaging:

- build docs index and figure pages
- expose a clean installable package surface
- surface confidence summaries early
- generate matrices automatically
- reconcile stale artifacts
- produce SVG panels and callout boards
- separate mechanism provenance from survey-map provenance
- align tags, release notes, and package metadata with the same repository milestone

Packaging is methodological here because it determines how readers infer credibility. A polished repo that hides uncertainty is less useful than a polished repo that exposes confidence and limitations.

> **Method Box 6: Packaging Has Two Audiences**
>
> The repository package is for readers who want the full research instrument: docs, figures, provenance, and generated artifacts. The Python package is for users who want the reusable code surface. Treating those as distinct outputs avoids bloated installs while keeping the full research record intact.

## Research Logic By Layer

### Layer A: Mechanism-Level Modules

These modules carry the strongest research burden in the repo. They are where local runnable claims live.

Their job is to answer questions like:

- what structural change is being modeled?
- what observable effect should follow?
- what local metric or trace captures that effect?

Examples:

- `transformer_basics` for causal attention structure
- `rag` for retrieval-grounded generation flow
- `position_bias_eval` for context-position effects
- `dpo_toy` for direct preference optimization logic
- `speculative_decoding_demo` for serving-time token acceptance behavior

### Layer B: Survey-Map Modules

These modules are not primarily runnable mechanisms. They are analytical compression layers over the mechanism surface.

Their job is to answer:

- how does the repo map onto the survey?
- what sections are covered?
- what bundle summaries exist?
- what publication views are defensible?

Examples:

- `paper_section_dashboard`
- `cross_section_summary`
- `adaptation_bundle_summary`
- `utilization_bundle_summary`

### Layer C: Resource/Reporting Modules

These modules provide orientation, inventory, provenance, and publication structure.

Their job is to answer:

- what is in the repo?
- how should it be read?
- what is donor-derived?
- what is computed?
- where are the artifacts?

Examples:

- model/resource registries
- provenance dashboards
- fidelity matrices
- publication assets

> **Method Box 7: Not Every Module Is Supposed To Be Equally Mechanistic**
>
> The repo is strongest when readers know which layer they are in. Mechanism modules explain behavior. Survey-map modules explain coverage. Resource/reporting modules explain organization and trust.

## What This Repository Can Legitimately Claim

This project can legitimately claim that it provides:

- a comprehensive local implementation map over a broad LLM survey surface
- dedicated module coverage for all tracked targets
- zero active generic fallback coverage
- provenance-aware separation between donor-derived, repo-authored, and computed-summary artifacts
- a publication-facing research surface with figures, matrices, and evidence tables
- a versioned package/release surface where `v0.3.0` aligns repository docs, release notes, and the `llm-survey` package metadata

It cannot legitimately claim:

- industrial-scale reproduction
- benchmark parity with frontier systems
- exhaustive coverage of every cited paper at equal fidelity
- direct comparability between all metrics without interpretation

## Why The Research Design Matters

The most important design choice in the repo is that it treats “understanding LLMs” as a layered reconstruction problem rather than a single-model problem.

That matters for education and for research hygiene:

- it prevents architecture from being confused with capability
- it prevents adaptation from being confused with pretraining
- it prevents benchmark packaging from being confused with mechanism
- it prevents polished dashboards from hiding low-fidelity assumptions

In short, the repo is designed to let a reader move from survey concept, to local mechanism, to report artifact, to publication summary, while keeping track of provenance and confidence at every step.

## Suggested Reading Path

For a first pass:

1. read [`README.md`](/Users/hifi/Downloads/LLM_Survey/README.md)
2. read [`docs/index.md`](/Users/hifi/Downloads/LLM_Survey/docs/index.md)
3. inspect [`docs/fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md)
4. inspect [`docs/mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
5. inspect [`docs/figures_and_tables.md`](/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md)

For a mechanism-first pass:

1. start in `src/modules/foundations/`
2. move to `pretraining`, `utilization`, and `evaluation`
3. inspect the matching `experiments/run_*_demo.py` scripts
4. inspect the corresponding JSON reports in `artifacts/generated/`

For a provenance-first pass:

1. inspect [`docs/module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md)
2. inspect [`docs/mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
3. inspect [`docs/survey_map_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/survey_map_provenance.md)
4. inspect [`docs/resource_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/resource_provenance.md)

## Final Interpretation

The deepest idea in this repository is that an LLM survey can be turned into a structured local research program if, and only if, three things are kept separate:

- conceptual coverage
- mechanism fidelity
- provenance confidence

This repo is the result of enforcing that separation all the way down:

- from modules
- to reports
- to tables
- to figures
- to documentation

That is its methodological contribution.
