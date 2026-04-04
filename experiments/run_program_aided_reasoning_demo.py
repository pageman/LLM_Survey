"""Local program-aided reasoning utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ProgramAidedReasoningDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ProgramAidedReasoningDemo().evaluate()
    report = build_report(
        experiment_id="program_aided_reasoning_demo",
        module="utilization.program_aided_reasoning_demo",
        metrics={
            "execution_gain": result["execution_gain"],
            "program_success": result["program_success"],
        },
        artifacts=result,
        notes=["Program-aided reasoning demo with synthesis, execution, and answer-reconciliation traces."],
    )
    write_report(report, output_dir / "program_aided_reasoning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
