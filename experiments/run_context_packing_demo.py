"""Local context-packing demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ContextPackingDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ContextPackingDemo().evaluate()
    report = build_report(
        experiment_id="context_packing_demo",
        module="utilization.context_packing_demo",
        metrics={
            "packed_efficiency": result["packed_efficiency"],
            "packing_gain": result["packing_gain"],
        },
        artifacts=result,
        notes=["Dedicated context-packing demo for multi-example batching efficiency."],
    )
    write_report(report, generated / "context_packing_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
