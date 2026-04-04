"""Dedicated cross-section summary with explicit section evidence and caveats."""

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
        section_rows = []
        for section, filenames in section_scores.items():
            scores = []
            evidence_rows = []
            for filename in filenames:
                payload = json.loads((reports_dir / filename).read_text())
                numeric_metrics = [float(value) for value in payload["metrics"].values() if isinstance(value, (int, float))]
                mean_metric = sum(numeric_metrics) / max(len(numeric_metrics), 1)
                scores.append(mean_metric)
                evidence_rows.append(
                    {
                        "report_file": filename,
                        "experiment_id": payload["experiment_id"],
                        "module": payload["module"],
                        "metric_count": len(numeric_metrics),
                        "mean_numeric_metric": round(mean_metric, 4),
                    }
                )
            sections[section] = round(sum(scores) / max(len(scores), 1), 4)
            section_rows.append(
                {
                    "section": section,
                    "score": sections[section],
                    "reports": evidence_rows,
                    "interpretation": "section-local summary only",
                }
            )
        best_section = max(sections, key=sections.get)
        return {
            "num_sections": len(sections),
            "best_section_score": sections[best_section],
            "best_section": best_section,
            "section_scores": sections,
            "section_rows": section_rows,
            "methodology_note": "Section scores average numeric metrics from a curated section-local report set and are not universal rankings across incompatible task families.",
        }
