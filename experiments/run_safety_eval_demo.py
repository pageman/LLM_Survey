"""Local safety evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import SafetyEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = SafetyEvaluator().evaluate()
    report = build_report(
        experiment_id="safety_eval_demo",
        module="evaluation.safety_eval",
        metrics={
            "refusal_rate": result["refusal_rate"],
            "jailbreak_success_rate": result["jailbreak_success_rate"],
        },
        artifacts=result,
        notes=["Toy refusal and jailbreak-resistance probe."],
    )
    write_report(report, output_dir / "safety_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
