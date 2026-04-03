"""Generate human-readable Markdown summaries from local report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_TARGETS = [
    "resources.public_model_registry",
    "resources.closed_model_registry",
    "resources.corpus_profile_demo",
    "resources.library_stack_matrix",
    "resources.framework_stack_matrix",
    "resources.dataset_license_audit",
    "resources.model_release_timeline",
    "foundations.seq2seq_basics",
    "foundations.rnn_lm",
    "foundations.lstm_lm",
    "foundations.transformer_basics",
    "pretraining.masked_lm_demo",
    "pretraining.prefix_decoder_demo",
    "pretraining.scaling_laws",
    "pretraining.causal_lm",
    "pretraining.multi_token_prediction",
    "pretraining.tokenizer_demo",
    "pretraining.data_mixture_toy",
    "pretraining.data_curriculum_demo",
    "pretraining.dedup_demo",
    "pretraining.contamination_demo",
    "pretraining.data_quality_filter_demo",
    "pretraining.repeated_data_scaling_demo",
    "pretraining.data_age_demo",
    "pretraining.domain_coverage_demo",
    "pretraining.toxicity_filter_demo",
    "pretraining.multilingual_data_demo",
    "pretraining.code_corpus_demo",
    "training.objective_mixture_demo",
    "training.optimizer_schedule_demo",
    "training.warmup_decay_demo",
    "training.batch_scaling_demo",
    "training.gradient_checkpointing_demo",
    "training.memory_partitioning_demo",
    "training.optimizer_ablation_dashboard",
    "architecture.encoder_decoder_demo",
    "architecture.prefix_lm_demo",
    "architecture.moe_demo",
    "architecture.bidirectional_encoder_demo",
    "architecture.multilingual_architecture_demo",
    "architecture.code_model_architecture_demo",
    "architecture.configuration_scaling_demo",
    "code_pretraining.program_synthesis_demo",
    "code_pretraining.nlp_as_code_demo",
    "systems.pipeline_parallelism",
    "systems.optimization_stability_demo",
    "systems.kv_cache_toy",
    "systems.inference_batching_demo",
    "systems.speculative_decoding_demo",
    "systems.kv_cache_fragmentation_demo",
    "utilization.retrieval",
    "utilization.rag",
    "utilization.icl_demo",
    "utilization.cot_prompting",
    "utilization.self_consistency_demo",
    "utilization.tool_use_stub",
    "utilization.planning_agent_demo",
    "utilization.example_selection_demo",
    "utilization.prompt_order_sensitivity_demo",
    "utilization.structured_prompting_demo",
    "utilization.least_to_most_demo",
    "utilization.react_demo",
    "utilization.world_model_planning_demo",
    "utilization.toolformer_style_demo",
    "utilization.program_aided_reasoning_demo",
    "utilization.scratchpad_demo",
    "utilization.context_packing_demo",
    "utilization.retrieval_selection_demo",
    "evaluation.long_context",
    "evaluation.position_bias_eval",
    "evaluation.benchmark_harness",
    "evaluation.calibration_eval",
    "evaluation.hallucination_checks",
    "evaluation.safety_eval",
    "evaluation.bias_eval",
    "evaluation.capability_suite_demo",
    "evaluation.code_eval_demo",
    "evaluation.math_reasoning_eval",
    "evaluation.embodied_planning_eval",
    "evaluation.multi_task_eval",
    "evaluation.formal_reasoning_eval",
    "evaluation.robustness_eval",
    "evaluation.out_of_distribution_eval",
    "evaluation.long_tail_behavior_eval",
    "evaluation.privacy_leakage_eval",
    "evaluation.truthfulness_eval",
    "evaluation.truthfulness_vs_helpfulness_eval",
    "evaluation.verifier_eval",
    "evaluation.jailbreak_transfer_eval",
    "evaluation.reward_model_overoptimization_demo",
    "adaptation.alignment_sft",
    "adaptation.finetuning",
    "adaptation.instruction_tuning",
    "adaptation.peft_lora",
    "adaptation.preference_tuning",
    "adaptation.reward_model_toy",
    "adaptation.dpo_toy",
    "adaptation.ppo_rlhf_toy",
    "adaptation.rejection_sampling_demo",
    "adaptation.constitutional_ai_demo",
    "adaptation.red_teaming_demo",
    "adaptation.instruction_data_construction_demo",
    "adaptation.memory_efficient_adaptation_demo",
    "adaptation.alignment_data_filter_demo",
    "adaptation.preference_data_quality_demo",
    "adaptation.constitution_sweep_demo",
    "applications.code_generation_demo",
    "applications.embodied_agent_stub",
    "applications.scientific_assistant_demo",
    "reporting.paper_section_dashboard",
    "reporting.module_provenance_dashboard",
    "reporting.fidelity_band_dashboard",
    "benchmark.cross_section_summary",
    "benchmark.risk_bundle_summary",
    "benchmark.adaptation_bundle_summary",
    "benchmark.utilization_bundle_summary",
    "multilingual.transfer_eval",
    "multilingual.prompting_demo",
    "code_generation_risk_eval",
    "retrieval_grounding_eval",
    "reasoning_faithfulness_eval",
    "safety_reasoning_tradeoff_demo",
    "capability_vs_alignment_tradeoff_demo",
    "memorization_vs_generalization_demo",
]


class DocsSummaryGenerator:
    """Build a Markdown scoreboard from generated JSON artifacts."""

    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)

    def _load(self, filename: str) -> dict[str, Any]:
        path = self.reports_dir / filename
        return json.loads(path.read_text())

    def compute_progress(self) -> dict[str, Any]:
        report_index = self._load("report_index_demo.json")
        available_modules = set(report_index["artifacts"]["modules"].keys())
        completed = [module for module in IMPLEMENTATION_TARGETS if module in available_modules]
        percentage = (len(completed) / len(IMPLEMENTATION_TARGETS)) * 100.0 if IMPLEMENTATION_TARGETS else 0.0
        return {
            "implemented": len(completed),
            "target": len(IMPLEMENTATION_TARGETS),
            "percentage": percentage,
            "completed_modules": completed,
            "remaining_modules": [module for module in IMPLEMENTATION_TARGETS if module not in available_modules],
        }

    def build_markdown(self) -> str:
        adaptation = self._load("adaptation_leaderboard_demo.json")
        benchmark = self._load("benchmark_harness_demo.json")
        report_index = self._load("report_index_demo.json")
        progress = self.compute_progress()

        top_gain = adaptation["artifacts"]["top_by_gain"][:5]
        top_efficiency = adaptation["artifacts"]["top_by_efficiency"][:5]
        top_loss = adaptation["artifacts"]["top_by_lowest_adapted_loss"][:5]

        lines = []
        lines.append("# LLM_Survey Scoreboard")
        lines.append("")
        lines.append("## Progress")
        lines.append("")
        lines.append(
            f"- NumPy-only implementation progress: **{progress['implemented']}/{progress['target']} "
            f"({progress['percentage']:.1f}%)**"
        )
        lines.append(f"- Generated reports indexed: **{report_index['metrics']['num_reports']}**")
        lines.append(f"- Compared metrics in benchmark harness: **{benchmark['metrics']['num_compared_metrics']}**")
        lines.append("")
        lines.append("## Adaptation Leaderboard")
        lines.append("")
        lines.append("### Top By Gain")
        lines.append("")
        lines.append("| Experiment | Gain | Adapted Loss |")
        lines.append("|---|---:|---:|")
        for row in top_gain:
            lines.append(f"| {row['experiment_id']} | {row['gain']:.4f} | {row['adapted_loss']:.4f} |")
        lines.append("")
        lines.append("### Top By Efficiency")
        lines.append("")
        lines.append("| Experiment | Efficiency | Trainable Fraction |")
        lines.append("|---|---:|---:|")
        for row in top_efficiency:
            efficiency = row["efficiency_score"]
            efficiency_text = "n/a" if efficiency is None else f"{efficiency:.4f}"
            fraction = row["trainable_fraction"]
            fraction_text = "n/a" if fraction is None else f"{fraction:.4f}"
            lines.append(f"| {row['experiment_id']} | {efficiency_text} | {fraction_text} |")
        lines.append("")
        lines.append("### Top By Lowest Adapted Loss")
        lines.append("")
        lines.append("| Experiment | Adapted Loss | Gain |")
        lines.append("|---|---:|---:|")
        for row in top_loss:
            lines.append(f"| {row['experiment_id']} | {row['adapted_loss']:.4f} | {row['gain']:.4f} |")
        lines.append("")
        lines.append("## Implemented Modules")
        lines.append("")
        for module in progress["completed_modules"]:
            lines.append(f"- `{module}`")
        lines.append("")
        lines.append("## Remaining Planned Modules")
        lines.append("")
        for module in progress["remaining_modules"]:
            lines.append(f"- `{module}`")
        lines.append("")
        return "\n".join(lines)
