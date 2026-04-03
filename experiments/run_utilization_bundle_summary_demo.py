"""Local utilization-bundle summary demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.benchmark import UtilizationBundleSummary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = UtilizationBundleSummary(generated).build()
    report = build_report(
        experiment_id="utilization_bundle_summary_demo",
        module="benchmark.utilization_bundle_summary",
        metrics={
            "num_reports": result["num_reports"],
            "best_utilization_score": result["best_utilization_score"],
        },
        artifacts=result,
        notes=["Dedicated utilization-bundle summary over retrieval, reasoning, and tool-use demos."],
    )
    write_report(report, generated / "utilization_bundle_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
