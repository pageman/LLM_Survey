"""Local truthfulness evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation.truthfulness_eval import TruthfulnessEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = TruthfulnessEvaluator().evaluate()
    report = build_report(
        experiment_id="truthfulness_eval_demo",
        module="evaluation.truthfulness_eval",
        metrics={
            "truthfulness_score": result["truthfulness_score"],
            "imitation_gap": result["imitation_gap"],
        },
        artifacts=result,
        notes=["Lite truthfulness evaluation against supported answers and imitative falsehoods."],
    )
    write_report(report, output_dir / "truthfulness_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
