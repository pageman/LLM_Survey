# Faithfulness Audit Plan

This document defines the next fidelity-hardening program for `LLM_Survey`. It is the working plan for turning the repository from "scope-complete and publication-packaged" into "more defensibly faithful to the mechanisms, derivations, evaluations, and evidence structure implied by the survey."

This plan assumes the repository remains:

- local-first
- NumPy-only
- educational rather than industrial-scale
- explicit about provenance, confidence, and limitations

## Why This Plan Exists

`LLM_Survey` is already strong on breadth:

- `124 / 124` tracked modules are present
- `100.0%` of tracked targets are dedicated modules
- `0.0%` generic-generator coverage remains
- publication docs, figures, provenance tables, and generated reports are in place

That is a completion milestone, not the end state.

The next quality problem is no longer scope. It is faithfulness.

In this repo, faithfulness means:

1. the module preserves the core mechanism of the topic
2. the module's derivation is traceable to donor code, local primitives, or explicit repo-authored logic
3. the evaluation matches the type of claim the module is making
4. the publication layer does not visually overclaim beyond the underlying artifact
5. limitations remain explicit

## Working Definitions

### Mechanistic Faithfulness

A module is mechanistically faithful when its core state transitions or optimization logic reflect the technical idea it names, rather than merely gesturing at the topic.

Examples:

- `transformer_basics` should expose causal masking, attention, and feed-forward structure
- `retrieval` should expose ranking and grounding behavior
- `dpo_toy` should expose chosen-vs-rejected preference updates
- `position_bias_eval` should expose evidence-position effects

### Derivational Faithfulness

A module is derivationally faithful when the implementation lineage is explicit:

- direct donor adaptation
- donor-derived local primitive reuse
- repo-authored mechanism with a clear local derivation
- computed summary only

### Evaluative Faithfulness

A module is evaluatively faithful when its metrics are appropriate to the claim it makes.

Bad pattern:

- using one generic scalar to stand in for a complex method without any trace or failure-mode structure

Better pattern:

- pair summary metrics with intermediate traces, item-level outcomes, or family-local comparisons

### Publication Faithfulness

A repository artifact is publication-faithful when it tells the truth about:

- what is mechanism-level
- what is survey-map only
- what is reporting/resource structure
- what is donor-derived
- what is repo-authored
- what is computed summary

## Current State Summary

Current strengths:

- full tracked coverage
- no generic fallback layer
- committed generated outputs
- explicit confidence tags
- evidence-backed provenance docs
- family-based benchmark structure
- coherent publication surface

Current remaining weaknesses:

- some mechanism demos are still trace-light or scalar-heavy
- several repo-authored modules have generic derivation notes that could be made more exact
- some evaluation modules still compress too much of the method into simplified scores
- some prompting, tool-use, and alignment demos need stronger explicit control-flow traces
- some survey-map/reporting artifacts are still structurally honest but analytically thin

## Audit Axes

Every module should be judged across these axes.

### Axis A: Mechanism Strength

Question:

- does the module expose the actual moving parts of the method?

Scoring:

- `A`: clear mechanism with meaningful internal state, transitions, or optimization
- `B`: simplified but still technically representative
- `C`: topic is present but method is shallow
- `D`: mostly naming/summary, not mechanism

### Axis B: Derivation Quality

Question:

- can a reader tell where the method came from and how the implementation was constructed?

Scoring:

- `A`: direct donor or exact local primitive linkage
- `B`: donor-adjacent conceptual linkage is clear
- `C`: repo-authored with plausible but weakly documented derivation
- `D`: opaque derivation

### Axis C: Evaluation Quality

Question:

- do the outputs and metrics match the type of claim the module is making?

Scoring:

- `A`: family-appropriate metrics plus traces or item-level structure
- `B`: family-appropriate metrics only
- `C`: informative but under-specified
- `D`: weak proxy or overcompressed score

### Axis D: Publication Risk

Question:

- is there a realistic chance that the docs or figures make the module look stronger than it is?

Scoring:

- `Low`
- `Medium`
- `High`

## Workstreams

## Workstream 1: Faithfulness Inventory

Deliverables:

- `docs/faithfulness_remediation_matrix.md`
- explicit module grading
- first-pass refactor priorities

Purpose:

- convert vague fidelity concerns into a tracked remediation program

## Workstream 2: Donor And Primitive Traceability

Goal:

- raise the number and precision of derivation paths for repo-authored mechanism modules

Tasks:

- link more modules to exact reused local primitives
- add module-level derivation notes
- distinguish direct donor adaptation from donor-inspired reuse
- expose derivation modes in docs or provenance outputs

High-value targets:

- prompting and planning modules
- evaluation modules with PDF-only evidence paths
- alignment modules with repo-authored optimization logic

## Workstream 3: Shared Mechanism Refactors

Goal:

- consolidate duplicated toy logic so modules differ only where the underlying method differs

Candidate shared cores:

- decoder-generation primitives
- retrieval/ranking primitives
- prompt-trace primitives
- preference-optimization primitives
- context-positioning primitives
- risk-probe primitives

Expected payoff:

- fewer inconsistent assumptions
- stronger provenance
- stronger tests

## Workstream 4: Evaluation Hardening

Goal:

- make module claims and metrics match more tightly

Tasks:

- strengthen item-level traces
- separate family-local from cross-family comparisons
- add failure-mode reporting for risk modules
- make benchmark aggregation more conservative

## Workstream 5: Publication Honesty Hardening

Goal:

- prevent polished surfaces from overstating low-fidelity artifacts

Tasks:

- ensure every figure has a clear interpretation boundary
- ensure confidence and evidence are visible before summary views
- make survey-map and reporting layers visibly distinct from mechanism layers

## Priority Bands

### Band P0: Immediate Faithfulness Risks

These should be addressed first because they are most likely to be overread by a technically serious reader.

- prompting and control-flow demos with limited traces
- adaptation demos with shallow optimization exposition
- scalar-heavy risk/evaluation demos
- crosscutting tradeoff demos that compress complex tensions into one narrow local score

### Band P1: Strong But Under-Explained Modules

These are already decent mechanisms but need better derivation transparency.

- donor-derived modules that should point more explicitly to reused local primitives
- repo-authored modules that currently cite only the survey PDF

### Band P2: Publication And Reporting Refinements

These matter for external trust but are lower urgency than mechanism fixes.

- stronger doc language
- richer figure captions
- expanded provenance fields

## Acceptance Criteria

This plan is complete when:

1. every mechanism-level module has an explicit faithfulness grade
2. high-risk `C` and `D` mechanism modules have concrete next actions
3. benchmark families are family-local by design, not merely by description
4. donor and local-primitive derivation notes are substantially sharper
5. publication docs distinguish mechanism, survey-map, and reporting artifacts at first glance
6. the remediation queue is tied to actual files, not only topic names

## Recommended Execution Order

1. complete the remediation matrix
2. refactor the highest-risk prompting, tool-use, alignment, and risk modules
3. deepen derivation/provenance for repo-authored mechanisms
4. tighten benchmark methodology
5. update publication docs and figures to reflect the stronger fidelity model
6. add fidelity-specific tests

## Non-Goals

This plan does not aim to:

- add new survey topics
- reproduce industrial-scale training
- claim benchmark parity with frontier systems
- replace educational lite implementations with heavy dependencies

The point is not to change the repo's nature. The point is to make its current nature more rigorous, more explicit, and more defensible.
