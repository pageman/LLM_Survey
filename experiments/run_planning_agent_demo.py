from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import PlanningAgentDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PlanningAgentDemo().evaluate()
    report = build_report("planning_agent_demo", "utilization.planning_agent_demo", {"success_rate": result["success_rate"]}, result, ["Toy planning-agent decomposition demo."])
    write_report(report, output_dir / "planning_agent_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
