"""Small benchmark harness for comparing local module reports side-by-side."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report_index import ReportIndex


DEFAULT_METRIC_SPECS = {
    "data_mixture_toy_demo": {
        "best_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "pretraining_mixture"},
    },
    "dedup_demo": {
        "best_quality_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "pretraining_curation"},
        "best_privacy_risk": {"goal": "min", "low": 0.0, "high": 1.0, "family": "pretraining_curation"},
    },
    "contamination_demo": {
        "max_inflation": {"goal": "min", "low": 0.0, "high": 1.0, "family": "pretraining_curation"},
    },
    "alignment_sft_demo": {
        "baseline_loss": {"goal": "min", "low": 0.0, "high": 25.0, "family": "adaptation_supervised"},
        "adapted_loss": {"goal": "min", "low": -1.0, "high": 25.0, "family": "adaptation_supervised"},
        "gain": {"goal": "max", "low": 0.0, "high": 5.0, "family": "adaptation_supervised"},
    },
    "finetuning_demo": {
        "baseline_loss": {"goal": "min", "low": 0.0, "high": 25.0, "family": "adaptation_supervised"},
        "adapted_loss": {"goal": "min", "low": -1.0, "high": 25.0, "family": "adaptation_supervised"},
        "gain": {"goal": "max", "low": 0.0, "high": 5.0, "family": "adaptation_supervised"},
    },
    "instruction_tuning_demo": {
        "baseline_loss": {"goal": "min", "low": 0.0, "high": 25.0, "family": "adaptation_supervised"},
        "adapted_loss": {"goal": "min", "low": -1.0, "high": 25.0, "family": "adaptation_supervised"},
        "gain": {"goal": "max", "low": 0.0, "high": 5.0, "family": "adaptation_supervised"},
    },
    "peft_lora_demo": {
        "baseline_loss": {"goal": "min", "low": 0.0, "high": 25.0, "family": "adaptation_parameter_efficient"},
        "adapted_loss": {"goal": "min", "low": -1.0, "high": 25.0, "family": "adaptation_parameter_efficient"},
        "gain": {"goal": "max", "low": 0.0, "high": 5.0, "family": "adaptation_parameter_efficient"},
        "trainable_fraction": {"goal": "min", "low": 0.0, "high": 1.0, "family": "adaptation_parameter_efficient"},
    },
    "preference_tuning_demo": {
        "baseline_loss": {"goal": "min", "low": 0.0, "high": 25.0, "family": "adaptation_preference"},
        "adapted_loss": {"goal": "min", "low": -1.0, "high": 25.0, "family": "adaptation_preference"},
        "gain": {"goal": "max", "low": 0.0, "high": 5.0, "family": "adaptation_preference"},
    },
    "retrieval_demo": {
        "dense_mrr": {"goal": "max", "low": 0.0, "high": 1.0, "family": "retrieval_ranking"},
        "bm25_mrr": {"goal": "max", "low": 0.0, "high": 1.0, "family": "retrieval_ranking"},
        "hybrid_mrr": {"goal": "max", "low": 0.0, "high": 1.0, "family": "retrieval_ranking"},
    },
    "retrieval_selection_demo": {
        "selection_confidence": {"goal": "max", "low": 0.0, "high": 1.0, "family": "retrieval_selection"},
        "hybrid_gain": {"goal": "max", "low": 0.0, "high": 1.0, "family": "retrieval_selection"},
    },
    "context_packing_demo": {
        "packed_efficiency": {"goal": "max", "low": 0.0, "high": 1.5, "family": "utilization_context"},
        "packing_gain": {"goal": "max", "low": 0.0, "high": 1.0, "family": "utilization_context"},
    },
    "tool_use_stub_demo": {
        "used_tool_rate": {"goal": "max", "low": 0.0, "high": 1.0, "family": "utilization_tool_use"},
    },
    "long_context_demo": {
        "best_edge_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_context_position"},
        "middle_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_context_position"},
        "edge_gap": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_context_position"},
    },
    "position_bias_eval_demo": {
        "edge_mean": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_context_position"},
        "middle_mean": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_context_position"},
        "edge_over_middle_ratio": {"goal": "min", "low": 1.0, "high": 2.0, "family": "evaluation_context_position"},
    },
    "calibration_eval_demo": {
        "ece": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_reliability"},
    },
    "hallucination_checks_demo": {
        "hallucination_rate": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
        "supported_rate": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
    },
    "truthfulness_eval_demo": {
        "truthfulness_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
        "imitation_gap": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
    },
    "truthfulness_vs_helpfulness_eval_demo": {
        "helpfulness_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
        "truthfulness_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
        "mean_gap": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_truth_grounding"},
    },
    "safety_eval_demo": {
        "refusal_rate": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_safety"},
        "jailbreak_success_rate": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_safety"},
    },
    "jailbreak_transfer_eval_demo": {
        "source_attack_rate": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_safety"},
        "transfer_attack_rate": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_safety"},
        "transfer_ratio": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_safety"},
    },
    "bias_eval_demo": {
        "stereotype_score": {"goal": "min", "low": 0.0, "high": 1.0, "family": "evaluation_fairness"},
        "fairness_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "evaluation_fairness"},
    },
    "optimizer_ablation_dashboard_demo": {
        "best_loss": {"goal": "min", "low": 0.0, "high": 3.0, "family": "training_optimization"},
        "loss_spread": {"goal": "min", "low": 0.0, "high": 2.0, "family": "training_optimization"},
    },
    "warmup_decay_demo": {
        "stability_score": {"goal": "max", "low": 0.0, "high": 1.0, "family": "training_optimization"},
        "final_gain": {"goal": "max", "low": 0.0, "high": 1.0, "family": "training_optimization"},
    },
    "batch_scaling_demo": {
        "throughput_gain": {"goal": "max", "low": 1.0, "high": 4.0, "family": "training_scaling"},
    },
    "gradient_checkpointing_demo": {
        "memory_reduction": {"goal": "max", "low": 0.0, "high": 1.0, "family": "training_memory"},
    },
    "memory_partitioning_demo": {
        "memory_saving": {"goal": "max", "low": 0.0, "high": 1.0, "family": "training_memory"},
    },
    "inference_batching_demo": {
        "latency_amortization": {"goal": "max", "low": 1.0, "high": 4.0, "family": "systems_serving"},
    },
    "speculative_decoding_demo": {
        "speedup": {"goal": "max", "low": 1.0, "high": 3.0, "family": "systems_serving"},
    },
    "kv_cache_fragmentation_demo": {
        "mean_fragmentation_penalty": {"goal": "min", "low": 0.0, "high": 1.0, "family": "systems_memory"},
        "worst_case_penalty": {"goal": "min", "low": 0.0, "high": 1.0, "family": "systems_memory"},
    },
}


class BenchmarkHarness:
    """Load generated reports and build a cross-module comparison table."""

    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)

    @staticmethod
    def normalize_value(value: float, spec: dict[str, Any]) -> float:
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        if high <= low:
            return 0.0
        if spec["goal"] == "max":
            score = (value - low) / (high - low)
        else:
            score = (high - value) / (high - low)
        return max(0.0, min(1.0, float(score)))

    def load_reports(self) -> list[dict[str, Any]]:
        index = ReportIndex(self.reports_dir).build()
        reports = []
        for row in index["reports"]:
            data = json.loads(Path(row["path"]).read_text())
            reports.append(data)
        return reports

    @staticmethod
    def family_group(family: str) -> str:
        if family.startswith("pretraining_"):
            return "pretraining"
        if family.startswith("training_"):
            return "training"
        if family.startswith("systems_"):
            return "systems"
        if family.startswith("retrieval_"):
            return "retrieval"
        if family.startswith("utilization_"):
            return "utilization"
        if family.startswith("evaluation_"):
            return "evaluation"
        if family.startswith("adaptation_"):
            return "adaptation"
        return "other"

    def compare(self, metric_specs: dict[str, dict[str, dict[str, str]]] | None = None) -> dict[str, Any]:
        metric_specs = metric_specs or DEFAULT_METRIC_SPECS
        reports = self.load_reports()

        indexed = {report["experiment_id"]: report for report in reports}
        comparisons = []
        by_experiment: dict[str, list[float]] = {}
        by_family: dict[str, list[float]] = {}
        by_family_group: dict[str, list[float]] = {}

        for experiment_id, metrics_config in metric_specs.items():
            report = indexed.get(experiment_id)
            if report is None:
                continue

            for metric_name, spec in metrics_config.items():
                if metric_name not in report["metrics"]:
                    continue
                value = report["metrics"][metric_name]
                normalized = self.normalize_value(float(value), spec)
                comparisons.append(
                    {
                        "experiment_id": experiment_id,
                        "module": report["module"],
                        "metric": metric_name,
                        "value": value,
                        "goal": spec["goal"],
                        "family": spec.get("family", "uncategorized"),
                        "family_group": self.family_group(spec.get("family", "uncategorized")),
                        "normalized_score": normalized,
                    }
                )
                by_experiment.setdefault(experiment_id, []).append(normalized)
                by_family.setdefault(spec.get("family", "uncategorized"), []).append(normalized)
                by_family_group.setdefault(self.family_group(spec.get("family", "uncategorized")), []).append(normalized)

        experiment_scores = [
            {
                "experiment_id": experiment_id,
                "module": indexed[experiment_id]["module"],
                "normalized_score": round(sum(scores) / len(scores), 4),
                "num_metrics": len(scores),
            }
            for experiment_id, scores in by_experiment.items()
        ]
        experiment_scores.sort(key=lambda row: row["normalized_score"], reverse=True)
        family_scores = [
            {
                "family": family,
                "family_group": self.family_group(family),
                "normalized_score": round(sum(scores) / len(scores), 4),
                "num_metrics": len(scores),
            }
            for family, scores in by_family.items()
        ]
        family_scores.sort(key=lambda row: row["normalized_score"], reverse=True)
        family_group_scores = [
            {
                "family_group": family_group,
                "normalized_score": round(sum(scores) / len(scores), 4),
                "num_metrics": len(scores),
            }
            for family_group, scores in by_family_group.items()
        ]
        family_group_scores.sort(key=lambda row: row["normalized_score"], reverse=True)

        return {
            "num_reports": len(reports),
            "num_compared_metrics": len(comparisons),
            "num_ranked_experiments": len(experiment_scores),
            "num_families": len(family_scores),
            "num_family_groups": len(family_group_scores),
            "comparisons": comparisons,
            "experiment_scores": experiment_scores,
            "family_scores": family_scores,
            "family_group_scores": family_group_scores,
        }
