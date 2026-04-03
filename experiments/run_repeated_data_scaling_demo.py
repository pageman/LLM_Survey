"""Local repeated-data scaling pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import RepeatedDataScalingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RepeatedDataScalingDemo().evaluate()
    report = build_report(
        experiment_id="repeated_data_scaling_demo",
        module="pretraining.repeated_data_scaling_demo",
        metrics={
            "best_repeat_ratio": result["best_repeat_ratio"],
            "best_validation_loss": result["best_validation_loss"],
            "overfit_gap": result["overfit_gap"],
        },
        artifacts=result,
        notes=["Lite repeated-data scaling demo with repeat-ratio versus validation-loss behavior."],
    )
    write_report(report, output_dir / "repeated_data_scaling_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
