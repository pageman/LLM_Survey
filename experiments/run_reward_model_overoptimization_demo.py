"""Local reward-model overoptimization evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import RewardModelOveroptimizationDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RewardModelOveroptimizationDemo().evaluate()
    report = build_report(
        experiment_id="reward_model_overoptimization_demo",
        module="evaluation.reward_model_overoptimization_demo",
        metrics={
            "reward_factuality_correlation": result["reward_factuality_correlation"],
            "overoptimization_gap": result["overoptimization_gap"],
        },
        artifacts=result,
        notes=["Lite reward-model overoptimization demo over reward versus factuality behavior."],
    )
    write_report(report, output_dir / "reward_model_overoptimization_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
