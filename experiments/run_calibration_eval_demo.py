"""Local calibration evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import CalibrationEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CalibrationEvaluator(seed=0).evaluate()
    report = build_report(
        experiment_id="calibration_eval_demo",
        module="evaluation.calibration_eval",
        metrics={"ece": result["ece"], "max_bin_gap": result["max_bin_gap"]},
        artifacts=result,
        notes=["Calibration evaluation with reliability bins and residual-gap reporting."],
    )
    write_report(report, output_dir / "calibration_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
