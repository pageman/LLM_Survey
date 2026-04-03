"""Local robustness evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import RobustnessEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RobustnessEvaluator().evaluate()
    report = build_report(
        experiment_id="robustness_eval_demo",
        module="evaluation.robustness_eval",
        metrics={"robustness_gap": result["robustness_gap"], "perturbed_score": result["perturbed_score"]},
        artifacts=result,
        notes=["Lite robustness demo under perturbations."],
    )
    write_report(report, output_dir / "robustness_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
