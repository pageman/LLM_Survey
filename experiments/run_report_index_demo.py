"""Build an index over all generated local reports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import ReportIndex


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    index = ReportIndex(generated).build()

    report = build_report(
        experiment_id="report_index_demo",
        module="evaluation.report_index",
        metrics={
            "num_reports": index["num_reports"],
            "num_modules": len(index["modules"]),
            "raw_num_reports": index["raw_num_reports"],
            "stale_report_count": index["stale_report_count"],
        },
        artifacts=index,
        notes=["Indexes canonical generated reports and surfaces stale duplicate artifacts separately."],
    )
    write_report(report, generated / "report_index_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
