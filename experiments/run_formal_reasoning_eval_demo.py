"""Local formal-reasoning evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import FormalReasoningEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = FormalReasoningEvaluator().evaluate()
    report = build_report(
        experiment_id="formal_reasoning_eval_demo",
        module="evaluation.formal_reasoning_eval",
        metrics={"proof_validity": result["proof_validity"], "formal_error_rate": result["formal_error_rate"]},
        artifacts=result,
        notes=["Lite formal reasoning demo over proof-step validity."],
    )
    write_report(report, output_dir / "formal_reasoning_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
