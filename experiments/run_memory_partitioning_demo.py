"""Local memory-partitioning training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import MemoryPartitioningDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MemoryPartitioningDemo().evaluate()
    report = build_report(
        experiment_id="memory_partitioning_demo",
        module="training.memory_partitioning_demo",
        metrics={"memory_saving": result["memory_saving"], "max_communication_cost": result["max_communication_cost"]},
        artifacts=result,
        notes=["Lite memory partitioning demo over shard count, memory savings, and communication cost."],
    )
    write_report(report, output_dir / "memory_partitioning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
