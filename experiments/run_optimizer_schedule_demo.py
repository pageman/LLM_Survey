"""Local optimizer-schedule training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import OptimizerScheduleDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = OptimizerScheduleDemo().evaluate()
    report = build_report(
        experiment_id="optimizer_schedule_demo",
        module="training.optimizer_schedule_demo",
        metrics={"final_loss": result["final_loss"], "schedule_gain": result["schedule_gain"]},
        artifacts=result,
        notes=["Lite optimizer schedule demo with warmup-decay behavior and loss tracking."],
    )
    write_report(report, output_dir / "optimizer_schedule_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
