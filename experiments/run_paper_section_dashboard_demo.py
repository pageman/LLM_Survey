"""Local paper-section dashboard demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.reporting import PaperSectionDashboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = PaperSectionDashboard(generated).build()
    report = build_report(
        experiment_id="paper_section_dashboard_demo",
        module="reporting.paper_section_dashboard",
        metrics={
            "num_sections": result["num_sections"],
            "mean_section_completion": result["mean_section_completion"],
        },
        artifacts=result,
        notes=["Dedicated paper-section dashboard using the report index and implementation target list."],
    )
    write_report(report, generated / "paper_section_dashboard_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
