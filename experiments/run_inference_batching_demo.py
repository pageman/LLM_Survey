"""Local inference-batching systems demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import InferenceBatchingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = InferenceBatchingDemo().evaluate()
    report = build_report(
        experiment_id="inference_batching_demo",
        module="systems.inference_batching_demo",
        metrics={"max_throughput": result["max_throughput"], "latency_amortization": result["latency_amortization"]},
        artifacts=result,
        notes=["Lite inference batching demo over latency amortization and throughput."],
    )
    write_report(report, output_dir / "inference_batching_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
