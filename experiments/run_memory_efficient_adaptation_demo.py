"""Local memory-efficient adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import MemoryEfficientAdaptationDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MemoryEfficientAdaptationDemo().evaluate()
    report = build_report(
        experiment_id="memory_efficient_adaptation_demo",
        module="adaptation.memory_efficient_adaptation_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
            "trainable_fraction": result["trainable_fraction"],
        },
        artifacts=result,
        notes=["Lite memory-efficient adaptation demo with small-trainable-fraction accounting."],
    )
    write_report(report, output_dir / "memory_efficient_adaptation_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
