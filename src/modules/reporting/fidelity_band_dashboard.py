"""Dedicated fidelity-band dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.modules.evaluation.docs_summary import IMPLEMENTATION_TARGETS


@dataclass
class FidelityBandDashboard:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        mechanism_prefixes = {
            "foundations",
            "pretraining",
            "training",
            "architecture",
            "code_pretraining",
            "systems",
            "utilization",
            "evaluation",
            "adaptation",
            "applications",
            "multilingual",
        }
        mechanism_top_level = {
            "code_generation_risk_eval",
            "retrieval_grounding_eval",
            "reasoning_faithfulness_eval",
            "safety_reasoning_tradeoff_demo",
            "capability_vs_alignment_tradeoff_demo",
            "memorization_vs_generalization_demo",
        }
        survey_map_prefixes = {"resources", "reporting", "benchmark"}
        mechanism = [
            name
            for name in IMPLEMENTATION_TARGETS
            if name in mechanism_top_level or name.split(".", 1)[0] in mechanism_prefixes
        ]
        survey_map = [name for name in IMPLEMENTATION_TARGETS if name.split(".", 1)[0] in survey_map_prefixes]
        return {
            "mechanism_level_count": len(mechanism),
            "survey_map_count": len(survey_map),
            "mechanism_level_fraction": round(len(mechanism) / len(IMPLEMENTATION_TARGETS), 4),
            "survey_map_fraction": round(len(survey_map) / len(IMPLEMENTATION_TARGETS), 4),
            "bands": {
                "mechanism_level": mechanism,
                "survey_map": survey_map,
            },
        }
