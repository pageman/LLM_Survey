from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.applications import EmbodiedAgentStub


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = EmbodiedAgentStub().evaluate()
    report = build_report("embodied_agent_stub_demo", "applications.embodied_agent_stub", {"task_success_rate": result["task_success_rate"]}, result, ["Toy embodied-agent application stub."])
    write_report(report, output_dir / "embodied_agent_stub_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
