# Figures And Tables

## Section Completion

![Paper section completion](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/paper_section_completion.svg)

| Section | Implemented | Target | Completion |
|---|---:|---:|---:|
| Adaptation | 16 | 16 | 100.0% |
| Applications | 3 | 3 | 100.0% |
| Architecture | 7 | 7 | 100.0% |
| Benchmarking | 4 | 4 | 100.0% |
| Code Pretraining | 2 | 2 | 100.0% |
| Evaluation | 22 | 22 | 100.0% |
| Foundations | 4 | 4 | 100.0% |
| Multilingual | 2 | 2 | 100.0% |
| Pre-training | 17 | 17 | 100.0% |
| Reporting | 3 | 3 | 100.0% |
| Resources | 7 | 7 | 100.0% |
| Systems | 6 | 6 | 100.0% |
| Training | 7 | 7 | 100.0% |
| Utilization | 18 | 18 | 100.0% |

## Benchmark Families

![Benchmark family scores](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/benchmark_family_scores.svg)

| Family | Score | Metrics |
|---|---:|---:|
| `pretraining_mixture` | 1.0000 | 1 |
| `retrieval_ranking` | 1.0000 | 3 |
| `training_scaling` | 1.0000 | 1 |
| `evaluation_reliability` | 0.9020 | 1 |
| `adaptation_preference` | 0.8942 | 3 |
| `evaluation_truth_grounding` | 0.8879 | 7 |
| `pretraining_curation` | 0.8591 | 3 |
| `systems_serving` | 0.7988 | 2 |
| `evaluation_fairness` | 0.7750 | 2 |
| `systems_memory` | 0.7512 | 2 |
| `utilization_tool_use` | 0.7500 | 1 |
| `adaptation_parameter_efficient` | 0.7024 | 4 |
| `evaluation_safety` | 0.6873 | 5 |
| `training_optimization` | 0.6851 | 3 |
| `utilization_context` | 0.5865 | 2 |
| `training_memory` | 0.5382 | 2 |
| `adaptation_supervised` | 0.4243 | 9 |
| `retrieval_selection` | 0.3130 | 2 |
| `evaluation_context_position` | 0.3060 | 6 |

## Benchmark Family Groups

| Group | Score | Metrics |
|---|---:|---:|
| `pretraining` | 0.8943 | 4 |
| `systems` | 0.7750 | 4 |
| `retrieval` | 0.7252 | 5 |
| `training` | 0.6886 | 6 |
| `evaluation` | 0.6638 | 21 |
| `utilization` | 0.6410 | 3 |
| `adaptation` | 0.5819 | 16 |

## Adaptation Trends

![Adaptation trends](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/adaptation_gain_trends.svg)

Caption: Gain, efficiency, and loss-improvement are normalized onto one shared axis so the leading adaptation demos can be compared without mixing raw scales.

## Retrieval Trends

![Retrieval trends](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/retrieval_slice_trends.svg)

Caption: Dense, sparse, and hybrid retrieval are compared with shared-scale MRR and recall series.

## Risk Trends

![Risk trends](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/risk_slice_trends.svg)

Caption: Beneficial and harmful signals are paired per evaluation slice to make tradeoffs visible at a glance.

## Publication Trend Panels

![Publication trend panels](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/trend_panels.svg)

Caption: Figure Set A uses a shared normalized scale and aligned framing so the panels behave like one publication figure.

## Trend Callout Boards

![Trend callout boards](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/trend_callouts.svg)

Caption: Figure Set B mirrors the plotted slices with exact values so the callout boards and panel figure can be read together.

## Top Normalized Benchmark Scores

| Experiment | Module | Score | Metrics |
|---|---|---:|---:|
| `data_mixture_toy_demo` | `pretraining.data_mixture_toy` | 1.0000 | 1 |
| `retrieval_demo` | `utilization.retrieval` | 1.0000 | 3 |
| `truthfulness_eval_demo` | `evaluation.truthfulness_eval` | 1.0000 | 2 |
| `batch_scaling_demo` | `training.batch_scaling_demo` | 1.0000 | 1 |
| `dedup_demo` | `pretraining.dedup_demo` | 0.9459 | 2 |
| `warmup_decay_demo` | `training.warmup_decay_demo` | 0.9271 | 1 |
| `calibration_eval_demo` | `evaluation.calibration_eval` | 0.9020 | 1 |
| `preference_tuning_demo` | `adaptation.preference_tuning` | 0.8942 | 3 |
| `speculative_decoding_demo` | `systems.speculative_decoding_demo` | 0.8851 | 1 |
| `safety_eval_demo` | `evaluation.safety_eval` | 0.8800 | 2 |

## Fidelity Split

![Fidelity band split](/Users/hifi/Downloads/LLM_Survey/artifacts/generated/figures/fidelity_band_split.svg)

## Artifact Tables

- Module matrix CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/module_matrix.csv`
- Benchmark experiment scores CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/benchmark_experiment_scores.csv`
- Benchmark family scores CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/benchmark_family_scores.csv`
- Benchmark family-group scores CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/benchmark_family_group_scores.csv`
- Section completion CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/paper_section_completion.csv`
- Survey-map provenance CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/survey_map_provenance.csv`
- Mechanism provenance CSV: `/Users/hifi/Downloads/LLM_Survey/artifacts/generated/tables/mechanism_provenance.csv`
