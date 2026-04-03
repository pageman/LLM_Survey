"""Local multitask evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import MultiTaskEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MultiTaskEvaluator().evaluate()
    report = build_report(
        experiment_id="multi_task_eval_demo",
        module="evaluation.multi_task_eval",
        metrics={"average_score": result["average_score"], "worst_task": result["worst_task"]},
        artifacts=result,
        notes=["Lite multitask evaluation over several toy task families."],
    )
    write_report(report, output_dir / "multi_task_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
