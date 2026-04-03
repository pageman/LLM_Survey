"""Local mixture-of-experts architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import MoEDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MoEDemo().evaluate()
    report = build_report(
        experiment_id="moe_demo",
        module="architecture.moe_demo",
        metrics={
            "load_balance": result["load_balance"],
            "expert_specialization": result["expert_specialization"],
        },
        artifacts=result,
        notes=["Lite mixture-of-experts demo with explicit routing, expert load, and specialization metrics."],
    )
    write_report(report, output_dir / "moe_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
