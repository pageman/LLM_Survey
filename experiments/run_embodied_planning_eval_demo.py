"""Local embodied planning evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import EmbodiedPlanningEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = EmbodiedPlanningEvaluator().evaluate()
    report = build_report(
        experiment_id="embodied_planning_eval_demo",
        module="evaluation.embodied_planning_eval",
        metrics={"success_rate": result["success_rate"], "path_consistency": result["path_consistency"]},
        artifacts=result,
        notes=["Lite embodied planning evaluation over toy episode success."],
    )
    write_report(report, output_dir / "embodied_planning_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
