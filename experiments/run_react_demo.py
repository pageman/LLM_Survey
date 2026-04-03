"""Local ReAct utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization.react_demo import ReActDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ReActDemo().evaluate()
    report = build_report(
        experiment_id="react_demo",
        module="utilization.react_demo",
        metrics={
            "task_success": result["task_success"],
            "grounded_reasoning_score": result["grounded_reasoning_score"],
            "avg_trace_length": result["avg_trace_length"],
        },
        artifacts=result,
        notes=["Lite ReAct demo with explicit reason-act-observe-answer traces."],
    )
    write_report(report, output_dir / "react_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
