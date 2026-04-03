"""Local gradient checkpointing training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import GradientCheckpointingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = GradientCheckpointingDemo().evaluate()
    report = build_report(
        experiment_id="gradient_checkpointing_demo",
        module="training.gradient_checkpointing_demo",
        metrics={
            "memory_reduction": result["memory_reduction"],
            "recompute_overhead": result["recompute_overhead"],
        },
        artifacts=result,
        notes=["Lite gradient checkpointing demo with memory-versus-recompute tradeoff."],
    )
    write_report(report, output_dir / "gradient_checkpointing_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
