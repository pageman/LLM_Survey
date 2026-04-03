"""Local cross-section summary demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.benchmark import CrossSectionSummary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = CrossSectionSummary(generated).build()
    report = build_report(
        experiment_id="cross_section_summary_demo",
        module="benchmark.cross_section_summary",
        metrics={
            "num_sections": result["num_sections"],
            "best_section_score": result["best_section_score"],
        },
        artifacts=result,
        notes=["Dedicated cross-section summary comparing pretraining, utilization, evaluation, and adaptation slices."],
    )
    write_report(report, generated / "cross_section_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
