"""Build a side-by-side benchmark summary from generated local reports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import BenchmarkHarness


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    harness = BenchmarkHarness(generated)
    summary = harness.compare()

    report = build_report(
        experiment_id="benchmark_harness_demo",
        module="evaluation.benchmark_harness",
        metrics={
            "num_reports": summary["num_reports"],
            "num_compared_metrics": summary["num_compared_metrics"],
            "num_ranked_experiments": summary["num_ranked_experiments"],
            "num_families": summary["num_families"],
            "num_family_groups": summary["num_family_groups"],
        },
        artifacts={
            "comparisons": summary["comparisons"],
            "experiment_scores": summary["experiment_scores"],
            "family_scores": summary["family_scores"],
            "family_group_scores": summary["family_group_scores"],
        },
        notes=["Compares selected metrics with normalized scoring so cross-demo summaries are more comparable."],
    )
    write_report(report, generated / "benchmark_harness_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
