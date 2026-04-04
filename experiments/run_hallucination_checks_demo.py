"""Local hallucination checks demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import HallucinationEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = HallucinationEvaluator().evaluate()
    report = build_report(
        experiment_id="hallucination_checks_demo",
        module="evaluation.hallucination_checks",
        metrics={
            "supported_rate": result["supported_rate"],
            "hallucination_rate": result["hallucination_rate"],
            "failure_mode_count": result["failure_mode_count"],
        },
        artifacts=result,
        notes=["Hallucination checks with supported answers and categorized unsupported failure modes."],
    )
    write_report(report, output_dir / "hallucination_checks_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
