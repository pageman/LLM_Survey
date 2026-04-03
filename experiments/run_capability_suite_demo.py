"""Local capability-suite evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import CapabilitySuiteDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CapabilitySuiteDemo().evaluate()
    report = build_report(
        experiment_id="capability_suite_demo",
        module="evaluation.capability_suite_demo",
        metrics={"suite_average": result["suite_average"], "suite_minimum": result["suite_minimum"]},
        artifacts=result,
        notes=["Lite capability-suite summary across multiple task families."],
    )
    write_report(report, output_dir / "capability_suite_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
