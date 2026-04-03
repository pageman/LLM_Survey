"""Local position-bias evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import PositionBiasEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = PositionBiasEvaluator(context_length=15)
    result = evaluator.evaluate()
    report = build_report(
        experiment_id="position_bias_eval_demo",
        module="evaluation.position_bias_eval",
        metrics={
            "edge_mean": result["edge_mean"],
            "middle_mean": result["middle_mean"],
            "edge_over_middle_ratio": result["edge_over_middle_ratio"],
        },
        artifacts=result,
        notes=["Summarizes edge preference relative to middle positions."],
    )
    write_report(report, output_dir / "position_bias_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
