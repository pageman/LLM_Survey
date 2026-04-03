# LLM_Survey Docs Index

This is the documentation landing page for the local `LLM_Survey` repository.

## Start Here

- Publication figures and tables: [`figures_and_tables.md`](/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md)
- Fidelity matrix: [`fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md)
- Survey-map provenance: [`survey_map_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/survey_map_provenance.md)
- Mechanism provenance: [`mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
- Resource provenance: [`resource_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/resource_provenance.md)
- Publication asset manifest: [`publication_assets_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/publication_assets_demo.json)

Artifact classes:

- `mechanism-level`: runnable code-first technical demos
- `survey-map`: evidence-backed analytical tables and cross-section summaries
- `resource/reporting`: provenance, inventory, and publication scaffolding

## Confidence Summary

Current publication-layer confidence split:

- `donor-derived`: `38`
- `repo-authored`: `80`
- `computed-summary`: `6`

Row-level confidence tags are available in [`module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md) and [`fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md).

## Core Documents

- Scoreboard: [`scoreboard.md`](/Users/hifi/Downloads/LLM_Survey/docs/scoreboard.md)
- Module matrix: [`module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md)
- Fidelity matrix: [`fidelity_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/fidelity_matrix.md)
- Figures and tables: [`figures_and_tables.md`](/Users/hifi/Downloads/LLM_Survey/docs/figures_and_tables.md)
- Mechanism provenance: [`mechanism_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/mechanism_provenance.md)
- Resource provenance: [`resource_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/resource_provenance.md)
- Survey-map provenance: [`survey_map_provenance.md`](/Users/hifi/Downloads/LLM_Survey/docs/survey_map_provenance.md)
- Paper coverage map: [`paper_map.md`](/Users/hifi/Downloads/LLM_Survey/docs/paper_map.md)
- Re-plan: [`full_survey_replan.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_replan.md)
- Full gap audit: [`full_survey_gap_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md)
- Publication checklist: [`publication_checklist.md`](/Users/hifi/Downloads/LLM_Survey/docs/publication_checklist.md)
- Limitations: [`limitations.md`](/Users/hifi/Downloads/LLM_Survey/docs/limitations.md)
- Release notes: [`release_notes.md`](/Users/hifi/Downloads/LLM_Survey/docs/release_notes.md)
- Reuse audit: [`reuse_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md)

## Generated Artifacts

- Scoreboard summary report: [`docs_summary_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/docs_summary_demo.json)
- Report index: [`report_index_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/report_index_demo.json)
- Benchmark summary: [`benchmark_harness_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/benchmark_harness_demo.json)
- Adaptation summary: [`adaptation_summary_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/adaptation_summary_demo.json)
- Adaptation leaderboard: [`adaptation_leaderboard_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/adaptation_leaderboard_demo.json)
- Publication assets: [`publication_assets_demo.json`](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/publication_assets_demo.json)

## Local Commands

```bash
python3 -m unittest discover -s tests
python3 experiments/run_all_local_demos.py
python3 experiments/run_docs_summary_demo.py
python3 experiments/run_publication_assets_demo.py
```
