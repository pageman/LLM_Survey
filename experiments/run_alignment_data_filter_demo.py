"""Local alignment-data filter adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import AlignmentDataFilterDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = AlignmentDataFilterDemo().evaluate()
    report = build_report(
        experiment_id="alignment_data_filter_demo",
        module="adaptation.alignment_data_filter_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts=result,
        notes=["Lite alignment-data filter demo over filtered versus raw supervision."],
    )
    write_report(report, output_dir / "alignment_data_filter_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
