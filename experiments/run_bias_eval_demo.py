"""Local bias evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import BiasEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = BiasEvaluator().evaluate()
    report = build_report(
        experiment_id="bias_eval_demo",
        module="evaluation.bias_eval",
        metrics={
            "stereotype_score": result["stereotype_score"],
            "fairness_score": result["fairness_score"],
        },
        artifacts=result,
        notes=["Toy bias/fairness probe for differential response behavior."],
    )
    write_report(report, output_dir / "bias_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
