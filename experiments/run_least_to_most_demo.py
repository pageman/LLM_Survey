"""Local least-to-most reasoning demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import LeastToMostDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = LeastToMostDemo().evaluate()
    report = build_report(
        experiment_id="least_to_most_demo",
        module="utilization.least_to_most_demo",
        metrics={"decomposition_gain": result["decomposition_gain"], "stepwise_success": result["stepwise_success"]},
        artifacts=result,
        notes=["Lite least-to-most demo comparing direct against decomposed reasoning."],
    )
    write_report(report, output_dir / "least_to_most_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
