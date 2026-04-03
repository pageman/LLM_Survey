"""Local preference-data quality adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import PreferenceDataQualityDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PreferenceDataQualityDemo().evaluate()
    report = build_report(
        experiment_id="preference_data_quality_demo",
        module="adaptation.preference_data_quality_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts=result,
        notes=["Lite preference-data quality demo over label-quality effects."],
    )
    write_report(report, output_dir / "preference_data_quality_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
