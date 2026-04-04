"""Local world-model planning utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import WorldModelPlanningDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = WorldModelPlanningDemo().evaluate()
    report = build_report(
        experiment_id="world_model_planning_demo",
        module="utilization.world_model_planning_demo",
        metrics={
            "plan_success": result["plan_success"],
            "state_value_gain": result["state_value_gain"],
            "replanning_rate": result["replanning_rate"],
        },
        artifacts=result,
        notes=["World-model planning demo with latent state rollout, prediction error, and replanning."],
    )
    write_report(report, output_dir / "world_model_planning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
