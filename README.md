# LLM_Survey

NumPy-only educational re-implementations of the major technical areas covered by *A Survey of Large Language Models* (`arXiv:2303.18223v19`).

This repository turns the survey into runnable, local-first artifacts:

- importable toy modules
- an installable Python package surface
- executable local demos
- machine-readable JSON reports
- human-readable scoreboards, provenance tables, and publication assets

## Publication Surface

Start here if you want the publication-facing outputs first:

- Docs landing page: [`docs/index.md`](/Users/hifi/Downloads/LLM_Survey/docs/index.md)
- Research narrative and method: [`docs/research_narrative_and_method.md`](/Users/hifi/Downloads/LLM_Survey/docs/research_narrative_and_method.md)
- Citation metadata: [`CITATION.cff`](https://github.com/pageman/LLM_Survey/blob/main/CITATION.cff)
- Figures and tables: [`docs/figures_and_tables.md`](/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md)
- Fidelity matrix: [`docs/fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md)
- Survey-map provenance: [`docs/survey_map_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/survey_map_provenance.md)
- Mechanism provenance: [`docs/mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
- Resource provenance: [`docs/resource_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/resource_provenance.md)
- Publication asset manifest: [`publication_assets_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/publication_assets_demo.json)

Artifact classes in this repo are intentionally distinct:

- `mechanism-level`: code-first educational demos and evaluations
- `survey-map`: analytical tables and bundle summaries across the implementation surface
- `resource/reporting`: provenance, inventory, and publication-facing organization layers

## Research Arc

The full long-form narrative and methodological arc is in [`docs/research_narrative_and_method.md`](/Users/hifi/Downloads/LLM_Survey/docs/research_narrative_and_method.md). The short version is:

This repository treats the survey as a target surface, not just a reading list. The research move is to translate survey breadth into a local implementation program with three separable layers: mechanism-level demos that preserve specific technical claims, survey-map modules that explain coverage and bundle comparisons, and resource/reporting modules that make provenance and confidence visible. The point is not benchmark-scale reproduction. The point is to make the logic of modern LLM systems inspectable end to end.

Methodologically, the repo moves in a strict sequence: explicit scope definition, donor-aware primitive extraction, dedicated module construction, canonical report emission, family-local comparison, provenance hardening, and publication packaging. That sequence matters because it prevents the usual failure modes of educational LLM repos: shallow topic coverage, opaque derivation, metric over-comparison, and polished dashboards that hide weak evidence. In this repo, coverage, fidelity, and provenance are intentionally separate research axes.

The current release surface reflects that same sequence. The repository now has a versioned research snapshot at `v0.3.0`, a GitHub release, and an installable `llm_survey` package that exposes the reusable code without bundling the full publication artifact tree into the wheel. The repo remains the canonical research surface; the package is the reusable code surface.

## Confidence Summary

Current publication-layer confidence split:

- `donor-derived`: `38`
- `repo-authored`: `80`
- `computed-summary`: `6`

This summary is generated from the canonical local provenance tables. Row-level confidence tags live in [`docs/module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md) and [`docs/fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md).

## Scope

The goal is not to reproduce the survey text. The goal is to reconstruct its main technical themes as small, inspectable implementations:

- foundations and architectures
- pre-training objectives and data issues
- adaptation and alignment
- utilization and prompting
- evaluation and risk analysis
- systems and efficiency
- selected application demos

Current tracked completion: **124/124 modules (100.0%)** against the repository's expanded paper-scope target list.
Current active generated artifacts are also reconciled: **0 stale duplicates** remain in [`artifacts/generated`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated).

Important caveat:

- this means the tracked paper-scope coverage program in this repo is complete
- it does not mean every subtopic or paper cited by the survey has been reproduced at equal fidelity

## Source

- Paper: *A Survey of Large Language Models*
- arXiv: <https://arxiv.org/abs/2303.18223>
- Local source used in this workspace: `/Users/hifi/Downloads/2303.18223v19.pdf`
- Survey resource repo: <https://github.com/RUCAIBox/LLMSurvey>

## License And Citation

This repository is released under the [`MIT License`](/Users/hifi/Downloads/LLM_Survey/LICENSE). Citation metadata for GitHub and downstream tooling is in [`CITATION.cff`](https://github.com/pageman/LLM_Survey/blob/main/CITATION.cff).

## What This Repo Contains

### Implemented Areas

- `foundations`: `seq2seq_basics`, `rnn_lm`, `lstm_lm`, `transformer_basics`
- `pretraining`: `masked_lm_demo`, `prefix_decoder_demo`, `scaling_laws`, `causal_lm`, `multi_token_prediction`, `tokenizer_demo`, `data_mixture_toy`, `data_curriculum_demo`, `dedup_demo`, `contamination_demo`, `data_quality_filter_demo`
- `systems`: `pipeline_parallelism`, `optimization_stability_demo`, `kv_cache_toy`
- `utilization`: `retrieval`, `rag`, `icl_demo`, `cot_prompting`, `self_consistency_demo`, `tool_use_stub`, `planning_agent_demo`
- `evaluation`: `long_context`, `position_bias_eval`, `benchmark_harness`, `calibration_eval`, `hallucination_checks`, `safety_eval`, `bias_eval`
- `adaptation`: `alignment_sft`, `finetuning`, `instruction_tuning`, `peft_lora`, `preference_tuning`, `reward_model_toy`
- `applications`: `code_generation_demo`, `embodied_agent_stub`, `scientific_assistant_demo`

### Output Layers

- canonical local experiment reports in `artifacts/generated/`
- repo-wide report index and normalized benchmark families
- adaptation summary and leaderboard
- provenance tables for survey-map, resource, and mechanism-level modules
- paper-style figure panels, including combined trend panels and CSV tables
- Markdown scoreboard in [`docs/scoreboard.md`](/Users/hifi/Downloads/LLM_Survey/docs/scoreboard.md)

## Layout

```text
LLM_Survey/
  README.md
  docs/
    scoreboard.md
    module_matrix.md
    paper_map.md
    full_survey_replan.md
    reuse_audit.md
  src/
    core/
    modules/
  experiments/
  tests/
  artifacts/generated/
```

## Quick Start

Use a Python interpreter with `numpy` available in the active environment.

Run the full local verification suite:

```bash
python3 -m unittest discover -s tests
```

Regenerate all local demo reports:

```bash
python3 experiments/run_all_local_demos.py
```

Refresh the human-readable scoreboard:

```bash
python3 experiments/run_docs_summary_demo.py
```

Refresh publication-facing figures, tables, and provenance docs:

```bash
python3 experiments/run_publication_assets_demo.py
```

## Packaging

The installable Python package is intended to expose the reusable code surface, not the full research artifact tree.

Current package/release surface:

- package name: `llm-survey`
- import name: `llm_survey`
- current tag: `v0.3.0`
- GitHub release: `v0.3.0`

Install locally in editable mode:

```bash
python3 -m pip install -e .
```

If you are packaging locally without network access, use:

```bash
python3 -m pip install --no-build-isolation -e .
```

Run the minimal package smoke test:

```bash
python3 scripts/package_smoke_test.py
```

Build release artifacts:

```bash
python3 -m build
python3 -m twine check dist/*
```

The published package surface is `llm_survey`, which wraps the repo's reusable `src` code while keeping docs, experiments, tests, and generated artifacts out of the wheel.

## Key Docs

- Docs landing page: [`docs/index.md`](/Users/hifi/Downloads/LLM_Survey/docs/index.md)
- Research narrative and method: [`docs/research_narrative_and_method.md`](/Users/hifi/Downloads/LLM_Survey/docs/research_narrative_and_method.md)
- Citation metadata: [`CITATION.cff`](https://github.com/pageman/LLM_Survey/blob/main/CITATION.cff)
- Figures and tables: [`docs/figures_and_tables.md`](/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md)
- Fidelity matrix: [`docs/fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md)
- Survey-map provenance: [`docs/survey_map_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/survey_map_provenance.md)
- Mechanism provenance: [`docs/mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
- Resource provenance: [`docs/resource_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/resource_provenance.md)
- Scoreboard: [`docs/scoreboard.md`](/Users/hifi/Downloads/LLM_Survey/docs/scoreboard.md)
- Module matrix: [`docs/module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md)
- Paper coverage map: [`docs/paper_map.md`](/Users/hifi/Downloads/LLM_Survey/docs/paper_map.md)
- Full-scope re-plan: [`docs/full_survey_replan.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_replan.md)
- Full gap audit: [`docs/full_survey_gap_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md)
- Publication checklist: [`docs/publication_checklist.md`](/Users/hifi/Downloads/LLM_Survey/docs/publication_checklist.md)
- Limitations: [`docs/limitations.md`](/Users/hifi/Downloads/LLM_Survey/docs/limitations.md)
- Release notes: [`docs/release_notes.md`](/Users/hifi/Downloads/LLM_Survey/docs/release_notes.md)
- Reuse audit: [`docs/reuse_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md)

## Verification Model

This repo is designed to run locally and openly:

- plain Python
- local `unittest`
- JSON artifact generation
- no dependency on GitHub Actions, CircleCI, or hosted CI

## Reuse Policy

This repo reuses ideas and donor implementations from the local/project source:

- `pageman/sutskever-30-implementations`

That donor repo is treated as a source of reusable educational building blocks, not something copied wholesale. Provenance and reuse intent are tracked in the docs.
