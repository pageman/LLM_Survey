# Full Survey Gap Audit

This document is the missing exhaustive audit requested for `LLM_Survey`.

It separates the survey-space into:

- `Explicit`: directly named or structurally foregrounded in the survey/repo index
- `Implicit`: required subtopics that are bundled inside explicit sections
- `Inferred`: implementation-relevant mechanics strongly implied by the cited literature
- `Extrapolated`: adjacent but still survey-consistent engineering layers needed for runnable reconstruction
- `Hidden`: cross-cutting concerns that are easy to miss when a repo only tracks top-level modules

The goal here is not to inflate scope with duplicates. The goal is to produce a de-duplicated, distinct, paper-scope re-plan for the next tracked target list.

## Audit Principles

- distinct topics only
- merge near-duplicates into one target when the implementation mechanism is the same
- separate topics only when the mechanism, evaluation logic, or survey role is materially different
- bias toward topics with broad peer validation and recurring emphasis across the survey ecosystem

## A. Explicit Topics Missing

These are directly supported by the survey structure and the RUCAIBox survey index.

### Resources

- `resources.public_model_registry`
- `resources.closed_model_registry`
- `resources.corpus_profile_demo`
- `resources.library_stack_matrix`
- `resources.framework_stack_matrix`

### Pre-Training Data

- `pretraining.repeated_data_scaling_demo`
- `pretraining.data_age_demo`
- `pretraining.domain_coverage_demo`
- `pretraining.toxicity_filter_demo`
- `pretraining.multilingual_data_demo`
- `pretraining.code_corpus_demo`

### Architectures

- `architecture.encoder_decoder_demo`
- `architecture.prefix_lm_demo`
- `architecture.moe_demo`
- `architecture.bidirectional_encoder_demo`
- `architecture.multilingual_architecture_demo`
- `architecture.code_model_architecture_demo`
- `architecture.configuration_scaling_demo`

### Training Algorithms

- `training.objective_mixture_demo`
- `training.optimizer_schedule_demo`
- `training.warmup_decay_demo`
- `training.batch_scaling_demo`
- `training.gradient_checkpointing_demo`
- `training.memory_partitioning_demo`

### Pre-Training On Code

- `code_pretraining.program_synthesis_demo`
- `code_pretraining.nlp_as_code_demo`

### Adaptation Tuning

- `adaptation.instruction_data_construction_demo`
- `adaptation.memory_efficient_adaptation_demo`

### Utilization

- `utilization.example_selection_demo`
- `utilization.prompt_order_sensitivity_demo`
- `utilization.structured_prompting_demo`
- `utilization.least_to_most_demo`
- `utilization.verifier_guided_reasoning_demo`
- `utilization.self_refine_demo`
- `utilization.react_demo`
- `utilization.world_model_planning_demo`

### Capacity Evaluation

- `evaluation.capability_suite_demo`
- `evaluation.code_eval_demo`
- `evaluation.math_reasoning_eval`
- `evaluation.embodied_planning_eval`
- `evaluation.multi_task_eval`
- `evaluation.formal_reasoning_eval`

## B. Implicit Topics Missing

These are not always promoted as separate top-level sections but are clearly required by the survey’s technical framing.

- `pretraining.dataset_mixture_weighting_demo`
- `pretraining.compute_data_balance_demo`
- `pretraining.token_budget_allocation_demo`
- `adaptation.alignment_data_filter_demo`
- `adaptation.preference_data_quality_demo`
- `utilization.retrieval_selection_demo`
- `utilization.context_packing_demo`
- `evaluation.robustness_eval`
- `evaluation.out_of_distribution_eval`
- `evaluation.long_tail_behavior_eval`
- `evaluation.privacy_leakage_eval`
- `evaluation.truthfulness_eval`

## C. Inferred Topics Missing

These are strongly implied by the combination of cited work and the implementation program needed to represent the survey faithfully.

- `adaptation.dpo_toy`
- `adaptation.ppo_rlhf_toy`
- `adaptation.rejection_sampling_demo`
- `adaptation.constitutional_ai_demo`
- `adaptation.red_teaming_demo`
- `utilization.toolformer_style_demo`
- `utilization.program_aided_reasoning_demo`
- `utilization.scratchpad_demo`
- `evaluation.verifier_eval`
- `evaluation.reward_model_overoptimization_demo`
- `systems.inference_batching_demo`
- `systems.speculative_decoding_demo`

## D. Extrapolated Topics Missing

These are not always cleanly separated in the survey text, but they are necessary if the repo is meant to be a serious educational implementation environment rather than a loose pile of demos.

- `reporting.paper_section_dashboard`
- `reporting.module_provenance_dashboard`
- `benchmark.cross_section_summary`
- `benchmark.risk_bundle_summary`
- `benchmark.adaptation_bundle_summary`
- `benchmark.utilization_bundle_summary`

## E. Hidden Cross-Cutting Topics Missing

These are the most likely to be missed because they sit between sections.

- `multilingual.transfer_eval`
- `multilingual.prompting_demo`
- `code_generation_risk_eval`
- `retrieval_grounding_eval`
- `reasoning_faithfulness_eval`
- `safety_reasoning_tradeoff_demo`
- `capability_vs_alignment_tradeoff_demo`
- `memorization_vs_generalization_demo`

## De-Duplicated High-Priority Next Targets

This is the recommended next-wave list after de-duplication across all five categories above.

### Wave 1: Data And Training Fidelity

- `pretraining.repeated_data_scaling_demo`
- `pretraining.data_age_demo`
- `pretraining.domain_coverage_demo`
- `pretraining.toxicity_filter_demo`
- `pretraining.multilingual_data_demo`
- `training.objective_mixture_demo`
- `training.optimizer_schedule_demo`
- `training.batch_scaling_demo`

### Wave 2: Architecture Fidelity

- `architecture.encoder_decoder_demo`
- `architecture.prefix_lm_demo`
- `architecture.moe_demo`
- `architecture.configuration_scaling_demo`

### Wave 3: Adaptation Fidelity

- `adaptation.dpo_toy`
- `adaptation.ppo_rlhf_toy`
- `adaptation.constitutional_ai_demo`
- `adaptation.red_teaming_demo`
- `adaptation.instruction_data_construction_demo`

### Wave 4: Utilization Fidelity

- `utilization.example_selection_demo`
- `utilization.prompt_order_sensitivity_demo`
- `utilization.structured_prompting_demo`
- `utilization.least_to_most_demo`
- `utilization.react_demo`
- `utilization.world_model_planning_demo`

### Wave 5: Evaluation Fidelity

- `evaluation.truthfulness_eval`
- `evaluation.robustness_eval`
- `evaluation.privacy_leakage_eval`
- `evaluation.verifier_eval`
- `evaluation.code_eval_demo`
- `evaluation.math_reasoning_eval`

## Re-Plan Summary

The repo’s earlier 41-module target list captured a strong runnable backbone. It did **not** fully capture the broader survey-space once resources, training algorithms, multilingual/code-specific pre-training, richer alignment methods, prompt-selection mechanics, and broader risk/capability evaluation are taken seriously.

This audit is the corrected, explicit, paper-scope gap analysis.
