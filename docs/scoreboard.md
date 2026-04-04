# LLM_Survey Scoreboard

## Progress

- NumPy-only implementation progress: **124/124 (100.0%)**
- Generated reports indexed: **147**
- Compared metrics in benchmark harness: **59**

## Adaptation Leaderboard

### Top By Gain

| Experiment | Gain | Adapted Loss |
|---|---:|---:|
| finetuning_demo | 3.1521 | 7.1075 |
| instruction_tuning_demo | 0.6889 | 9.5725 |
| peft_lora_demo | nan | nan |
| preference_tuning_demo | 4.1090 | -0.8121 |
| alignment_sft_demo | 1.2605 | 20.0514 |

### Top By Efficiency

| Experiment | Efficiency | Trainable Fraction |
|---|---:|---:|
| finetuning_demo | 3.1521 | n/a |
| instruction_tuning_demo | 0.6889 | n/a |
| peft_lora_demo | nan | 0.4286 |
| preference_tuning_demo | 4.1090 | n/a |
| alignment_sft_demo | 1.2605 | n/a |

### Top By Lowest Adapted Loss

| Experiment | Adapted Loss | Gain |
|---|---:|---:|
| preference_tuning_demo | -0.8121 | 4.1090 |
| finetuning_demo | 7.1075 | 3.1521 |
| instruction_tuning_demo | 9.5725 | 0.6889 |
| alignment_sft_demo | 20.0514 | 1.2605 |
| peft_lora_demo | nan | nan |

## Implemented Modules

- `resources.public_model_registry`
- `resources.closed_model_registry`
- `resources.corpus_profile_demo`
- `resources.library_stack_matrix`
- `resources.framework_stack_matrix`
- `resources.dataset_license_audit`
- `resources.model_release_timeline`
- `foundations.seq2seq_basics`
- `foundations.rnn_lm`
- `foundations.lstm_lm`
- `foundations.transformer_basics`
- `pretraining.masked_lm_demo`
- `pretraining.prefix_decoder_demo`
- `pretraining.scaling_laws`
- `pretraining.causal_lm`
- `pretraining.multi_token_prediction`
- `pretraining.tokenizer_demo`
- `pretraining.data_mixture_toy`
- `pretraining.data_curriculum_demo`
- `pretraining.dedup_demo`
- `pretraining.contamination_demo`
- `pretraining.data_quality_filter_demo`
- `pretraining.repeated_data_scaling_demo`
- `pretraining.data_age_demo`
- `pretraining.domain_coverage_demo`
- `pretraining.toxicity_filter_demo`
- `pretraining.multilingual_data_demo`
- `pretraining.code_corpus_demo`
- `training.objective_mixture_demo`
- `training.optimizer_schedule_demo`
- `training.warmup_decay_demo`
- `training.batch_scaling_demo`
- `training.gradient_checkpointing_demo`
- `training.memory_partitioning_demo`
- `training.optimizer_ablation_dashboard`
- `architecture.encoder_decoder_demo`
- `architecture.prefix_lm_demo`
- `architecture.moe_demo`
- `architecture.bidirectional_encoder_demo`
- `architecture.multilingual_architecture_demo`
- `architecture.code_model_architecture_demo`
- `architecture.configuration_scaling_demo`
- `code_pretraining.program_synthesis_demo`
- `code_pretraining.nlp_as_code_demo`
- `systems.pipeline_parallelism`
- `systems.optimization_stability_demo`
- `systems.kv_cache_toy`
- `systems.inference_batching_demo`
- `systems.speculative_decoding_demo`
- `systems.kv_cache_fragmentation_demo`
- `utilization.retrieval`
- `utilization.rag`
- `utilization.icl_demo`
- `utilization.cot_prompting`
- `utilization.self_consistency_demo`
- `utilization.tool_use_stub`
- `utilization.planning_agent_demo`
- `utilization.example_selection_demo`
- `utilization.prompt_order_sensitivity_demo`
- `utilization.structured_prompting_demo`
- `utilization.least_to_most_demo`
- `utilization.react_demo`
- `utilization.world_model_planning_demo`
- `utilization.toolformer_style_demo`
- `utilization.program_aided_reasoning_demo`
- `utilization.scratchpad_demo`
- `utilization.context_packing_demo`
- `utilization.retrieval_selection_demo`
- `evaluation.long_context`
- `evaluation.position_bias_eval`
- `evaluation.benchmark_harness`
- `evaluation.calibration_eval`
- `evaluation.hallucination_checks`
- `evaluation.safety_eval`
- `evaluation.bias_eval`
- `evaluation.capability_suite_demo`
- `evaluation.code_eval_demo`
- `evaluation.math_reasoning_eval`
- `evaluation.embodied_planning_eval`
- `evaluation.multi_task_eval`
- `evaluation.formal_reasoning_eval`
- `evaluation.robustness_eval`
- `evaluation.out_of_distribution_eval`
- `evaluation.long_tail_behavior_eval`
- `evaluation.privacy_leakage_eval`
- `evaluation.truthfulness_eval`
- `evaluation.truthfulness_vs_helpfulness_eval`
- `evaluation.verifier_eval`
- `evaluation.jailbreak_transfer_eval`
- `evaluation.reward_model_overoptimization_demo`
- `adaptation.alignment_sft`
- `adaptation.finetuning`
- `adaptation.instruction_tuning`
- `adaptation.peft_lora`
- `adaptation.preference_tuning`
- `adaptation.reward_model_toy`
- `adaptation.dpo_toy`
- `adaptation.ppo_rlhf_toy`
- `adaptation.rejection_sampling_demo`
- `adaptation.constitutional_ai_demo`
- `adaptation.red_teaming_demo`
- `adaptation.instruction_data_construction_demo`
- `adaptation.memory_efficient_adaptation_demo`
- `adaptation.alignment_data_filter_demo`
- `adaptation.preference_data_quality_demo`
- `adaptation.constitution_sweep_demo`
- `applications.code_generation_demo`
- `applications.embodied_agent_stub`
- `applications.scientific_assistant_demo`
- `reporting.paper_section_dashboard`
- `reporting.module_provenance_dashboard`
- `reporting.fidelity_band_dashboard`
- `benchmark.cross_section_summary`
- `benchmark.risk_bundle_summary`
- `benchmark.adaptation_bundle_summary`
- `benchmark.utilization_bundle_summary`
- `multilingual.transfer_eval`
- `multilingual.prompting_demo`
- `code_generation_risk_eval`
- `retrieval_grounding_eval`
- `reasoning_faithfulness_eval`
- `safety_reasoning_tradeoff_demo`
- `capability_vs_alignment_tradeoff_demo`
- `memorization_vs_generalization_demo`

## Remaining Planned Modules

