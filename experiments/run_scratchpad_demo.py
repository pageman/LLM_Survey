"""Local scratchpad utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ScratchpadDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ScratchpadDemo().evaluate()
    report = build_report(
        experiment_id="scratchpad_demo",
        module="utilization.scratchpad_demo",
        metrics={"scratchpad_gain": result["scratchpad_gain"], "trace_consistency": result["trace_consistency"]},
        artifacts=result,
        notes=["Lite scratchpad demo comparing plain reasoning against scratchpad traces."],
    )
    write_report(report, output_dir / "scratchpad_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
