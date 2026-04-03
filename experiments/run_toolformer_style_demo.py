"""Local Toolformer-style utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ToolformerStyleDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ToolformerStyleDemo().evaluate()
    report = build_report(
        experiment_id="toolformer_style_demo",
        module="utilization.toolformer_style_demo",
        metrics={
            "tool_call_rate": result["tool_call_rate"],
            "tool_use_gain": result["tool_use_gain"],
        },
        artifacts=result,
        notes=["Lite Toolformer-style demo with self-inserted tool calls and answer-gain accounting."],
    )
    write_report(report, output_dir / "toolformer_style_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
