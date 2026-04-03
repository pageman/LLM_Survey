"""Local adaptation-bundle summary demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.benchmark import AdaptationBundleSummary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = AdaptationBundleSummary(generated).build()
    report = build_report(
        experiment_id="adaptation_bundle_summary_demo",
        module="benchmark.adaptation_bundle_summary",
        metrics={
            "num_reports": result["num_reports"],
            "mean_gain": result["mean_gain"],
            "best_gain": result["best_gain"],
        },
        artifacts=result,
        notes=["Dedicated adaptation-bundle summary over core adaptation demos."],
    )
    write_report(report, generated / "adaptation_bundle_summary_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
