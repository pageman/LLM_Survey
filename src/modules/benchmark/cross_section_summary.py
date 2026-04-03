"""Dedicated cross-section summary over generated reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrossSectionSummary:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        reports_dir = Path(self.reports_dir)
        section_scores = {
            "pretraining": ["scaling_laws_demo.json", "data_mixture_toy_demo.json", "masked_lm_demo.json"],
            "utilization": ["retrieval_demo.json", "rag_demo.json", "react_demo.json"],
            "evaluation": ["truthfulness_eval_demo.json", "robustness_eval_demo.json", "capability_suite_demo.json"],
            "adaptation": ["finetuning_demo.json", "dpo_toy_demo.json", "peft_lora_demo.json"],
        }
        sections = {}
        for section, filenames in section_scores.items():
            scores = []
            for filename in filenames:
                payload = json.loads((reports_dir / filename).read_text())
                numeric_metrics = [float(value) for value in payload["metrics"].values() if isinstance(value, (int, float))]
                scores.append(sum(numeric_metrics) / max(len(numeric_metrics), 1))
            sections[section] = round(sum(scores) / max(len(scores), 1), 4)
        best_section = max(sections, key=sections.get)
        return {
            "num_sections": len(sections),
            "best_section_score": sections[best_section],
            "best_section": best_section,
            "section_scores": sections,
        }
