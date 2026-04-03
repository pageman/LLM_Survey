"""Local math-reasoning evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import MathReasoningEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MathReasoningEvaluator().evaluate()
    report = build_report(
        experiment_id="math_reasoning_eval_demo",
        module="evaluation.math_reasoning_eval",
        metrics={"accuracy": result["accuracy"], "error_rate": result["error_rate"]},
        artifacts=result,
        notes=["Lite math reasoning evaluation over small exact-answer tasks."],
    )
    write_report(report, output_dir / "math_reasoning_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
