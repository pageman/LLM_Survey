"""Local tool-use stub demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ToolUseStub


def main() -> None:
    output_dir = Path("artifacts/generated")
    stub = ToolUseStub()
    queries = [
        "what is the weather in manila",
        "search for large language model papers",
        "calculate 12 times 7",
        "write a short poem",
    ]
    routed = [stub.route(query) for query in queries]
    used_rate = sum(1 for item in routed if item["used_tool"]) / len(routed)

    report = build_report(
        experiment_id="tool_use_stub_demo",
        module="utilization.tool_use_stub",
        metrics={"used_tool_rate": used_rate},
        artifacts={"routes": routed},
        notes=["Minimal tool routing stub for utilization-layer coverage."],
    )
    write_report(report, output_dir / "tool_use_stub_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
