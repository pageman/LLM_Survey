"""Local deduplication demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DeduplicationExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DeduplicationExperiment(seed=1).evaluate()
    report = build_report(
        experiment_id="dedup_demo",
        module="pretraining.dedup_demo",
        metrics={
            "best_quality_score": max(result["quality_scores"]),
            "best_privacy_risk": min(result["privacy_risks"]),
        },
        artifacts=result,
        notes=["Toy repeated-data study for quality and privacy tradeoffs."],
    )
    write_report(report, output_dir / "dedup_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
