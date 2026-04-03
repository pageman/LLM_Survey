"""Local instruction-data construction adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import InstructionDataConstructionDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = InstructionDataConstructionDemo().evaluate()
    report = build_report(
        experiment_id="instruction_data_construction_demo",
        module="adaptation.instruction_data_construction_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts=result,
        notes=["Lite instruction-data construction demo over diversity-quality tradeoffs."],
    )
    write_report(report, output_dir / "instruction_data_construction_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
