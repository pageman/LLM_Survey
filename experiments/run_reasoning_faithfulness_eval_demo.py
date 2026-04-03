"""Local reasoning-faithfulness evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import ReasoningFaithfulnessEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ReasoningFaithfulnessEvaluator().evaluate()
    report = build_report(
        experiment_id="reasoning_faithfulness_eval_demo",
        module="reasoning_faithfulness_eval",
        metrics={
            "truthfulness_score": result["truthfulness_score"],
            "faithfulness_gap": result["faithfulness_gap"],
        },
        artifacts=result,
        notes=["Lite reasoning-faithfulness demo comparing answer accuracy against trace faithfulness."],
    )
    write_report(report, output_dir / "reasoning_faithfulness_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
