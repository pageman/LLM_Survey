"""Dedicated dashboard summarizing coverage by paper section."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.modules.evaluation.docs_summary import IMPLEMENTATION_TARGETS
from src.modules.evaluation.report_index import ReportIndex


SECTION_PREFIXES = {
    "resources": "Resources",
    "foundations": "Foundations",
    "pretraining": "Pre-training",
    "training": "Training",
    "architecture": "Architecture",
    "code_pretraining": "Code Pretraining",
    "systems": "Systems",
    "utilization": "Utilization",
    "evaluation": "Evaluation",
    "adaptation": "Adaptation",
    "applications": "Applications",
    "reporting": "Reporting",
    "benchmark": "Benchmarking",
    "multilingual": "Multilingual",
}


@dataclass
class PaperSectionDashboard:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        indexed = ReportIndex(self.reports_dir).build()["modules"]
        sections = []
        for prefix, label in SECTION_PREFIXES.items():
            targets = [name for name in IMPLEMENTATION_TARGETS if name.startswith(f"{prefix}.")]
            completed = [name for name in targets if name in indexed]
            percentage = (len(completed) / len(targets) * 100.0) if targets else 0.0
            sections.append(
                {
                    "section": label,
                    "prefix": prefix,
                    "implemented": len(completed),
                    "target": len(targets),
                    "percentage": round(percentage, 1),
                }
            )
        sections.sort(key=lambda row: row["prefix"])
        return {
            "num_sections": len(sections),
            "mean_section_completion": round(
                sum(section["percentage"] for section in sections) / max(len(sections), 1), 1
            ),
            "sections": sections,
        }
