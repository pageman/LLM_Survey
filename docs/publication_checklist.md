# Publication Checklist

This checklist distinguishes what is already present in `LLM_Survey`, what is still missing for broader survey fidelity, and what would improve publication quality without changing the current report schema.

## Must-Haves

### Present

- Clear top-level overview in [`README.md`](/Users/hifi/Downloads/LLM_Survey/README.md)
- Human-readable scoreboard in [`scoreboard.md`](/Users/hifi/Downloads/LLM_Survey/docs/scoreboard.md)
- Module coverage matrix in [`module_matrix.md`](/Users/hifi/Downloads/LLM_Survey/docs/module_matrix.md)
- Coverage/replan docs in [`paper_map.md`](/Users/hifi/Downloads/LLM_Survey/docs/paper_map.md), [`full_survey_replan.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_replan.md), and [`full_survey_gap_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/full_survey_gap_audit.md)
- Reuse/provenance notes in [`reuse_audit.md`](/Users/hifi/Downloads/LLM_Survey/docs/reuse_audit.md)
- Local verification path with `unittest`
- Local-first experiment runner and generated JSON reports
- Consistent report schema for demos and summaries

### Missing

- Stronger paper-faithful implementations for the weakest lite paper-scope modules
- A more explicit citation / attribution section in the README for donor implementations

## Good-To-Haves

- Per-module narrative docs explaining how each toy demo maps back to the survey literature
- A static index page that links the scoreboard, module matrix, and major artifacts in one place
- A compact publication changelog summarizing milestone waves
- A packaged local command wrapper such as a simple `Makefile` or shell entrypoint

## Fidelity Priorities

If the goal is stronger scholarly fidelity rather than broader packaging, the highest-value missing blocks remain:

- resources/model and corpus registries
- richer pre-training data and training dynamics
- broader architecture coverage such as MoE and bidirectional/prefix variants
- richer utilization patterns such as ReAct, least-to-most, toolformer-style use, and prompt selection
- broader evaluation layers such as truthfulness, verifier-based reasoning, privacy leakage, and robustness
- stronger alignment topics such as DPO, PPO-style RLHF, constitutional methods, and red teaming

## Recommended Next Packaging Step

The repo is publication-packaged enough for a strong internal/public draft. The next packaging improvement should be:

- add a `docs/limitations.md` page that states which modules are faithful demos versus scope-completion placeholders

## Recommended Next Fidelity Step

The next highest-value fidelity wave should start with:

- `pretraining.repeated_data_scaling_demo`
- `pretraining.data_age_demo`
- `pretraining.domain_coverage_demo`
- `architecture.encoder_decoder_demo`
- `utilization.react_demo`
- `evaluation.truthfulness_eval`
- `adaptation.dpo_toy`
