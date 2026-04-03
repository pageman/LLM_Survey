"""Local OOD evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import OutOfDistributionEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = OutOfDistributionEvaluator().evaluate()
    report = build_report(
        experiment_id="out_of_distribution_eval_demo",
        module="evaluation.out_of_distribution_eval",
        metrics={"ood_gap": result["ood_gap"], "ood_score": result["ood_score"]},
        artifacts=result,
        notes=["Lite OOD generalization demo."],
    )
    write_report(report, output_dir / "out_of_distribution_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
