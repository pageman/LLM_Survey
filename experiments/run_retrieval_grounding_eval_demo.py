"""Local retrieval-grounding evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import RetrievalGroundingEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RetrievalGroundingEvaluator().evaluate()
    report = build_report(
        experiment_id="retrieval_grounding_eval_demo",
        module="retrieval_grounding_eval",
        metrics={"grounding_score": result["grounding_score"], "support_floor": result["support_floor"]},
        artifacts=result,
        notes=["Dedicated retrieval-grounding demo connecting support rate and hallucination pressure."],
    )
    write_report(report, output_dir / "retrieval_grounding_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
