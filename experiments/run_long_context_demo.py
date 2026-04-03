"""Local long-context demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import LongContextEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = LongContextEvaluator(context_length=15)
    result = evaluator.evaluate()
    report = build_report(
        experiment_id="long_context_demo",
        module="evaluation.long_context",
        metrics={
            "best_edge_score": result["best_edge_score"],
            "middle_score": result["middle_score"],
            "edge_gap": result["edge_gap"],
        },
        artifacts=result,
        notes=["Toy U-shaped long-context performance profile inspired by Lost in the Middle."],
    )
    write_report(report, output_dir / "long_context_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
