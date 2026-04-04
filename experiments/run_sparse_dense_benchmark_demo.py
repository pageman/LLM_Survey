"""Run the sparse-vs-dense benchmark slice demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import SparseDenseBenchmarkDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = SparseDenseBenchmarkDemo().evaluate()
    report = build_report(
        experiment_id="sparse_dense_benchmark_demo",
        module="systems.sparse_dense_benchmark_demo",
        metrics={
            "mean_efficiency": result["mean_efficiency"],
            "mean_approximation_gap": result["mean_approximation_gap"],
            "best_efficiency": result["best_efficiency"],
        },
        artifacts=result,
        notes=["Benchmark-style comparison across block-sparse and sliding-window sparse attention variants."],
    )
    write_report(report, generated / "sparse_dense_benchmark_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
