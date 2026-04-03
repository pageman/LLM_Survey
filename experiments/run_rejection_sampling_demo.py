"""Local rejection-sampling adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import RejectionSamplingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RejectionSamplingDemo().evaluate()
    report = build_report(
        experiment_id="rejection_sampling_demo",
        module="adaptation.rejection_sampling_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts=result,
        notes=["Lite rejection-sampling adaptation demo over candidate-reward filtering."],
    )
    write_report(report, output_dir / "rejection_sampling_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
