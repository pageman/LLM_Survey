"""Local batch-scaling training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import BatchScalingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = BatchScalingDemo().evaluate()
    report = build_report(
        experiment_id="batch_scaling_demo",
        module="training.batch_scaling_demo",
        metrics={"best_batch_size": result["best_batch_size"], "throughput_gain": result["throughput_gain"]},
        artifacts=result,
        notes=["Lite batch-scaling demo over throughput and quality tradeoffs."],
    )
    write_report(report, output_dir / "batch_scaling_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
