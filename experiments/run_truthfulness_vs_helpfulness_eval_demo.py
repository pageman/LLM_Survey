"""Local truthfulness-vs-helpfulness evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import TruthfulnessHelpfulnessEvaluator


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = TruthfulnessHelpfulnessEvaluator().evaluate()
    report = build_report(
        experiment_id="truthfulness_vs_helpfulness_eval_demo",
        module="evaluation.truthfulness_vs_helpfulness_eval",
        metrics={
            "helpfulness_score": result["helpfulness_score"],
            "truthfulness_score": result["truthfulness_score"],
            "mean_gap": result["mean_gap"],
        },
        artifacts=result,
        notes=["Dedicated evaluation of helpfulness against truthfulness retention."],
    )
    write_report(report, generated / "truthfulness_vs_helpfulness_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
