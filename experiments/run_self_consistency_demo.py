from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import SelfConsistencyDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = SelfConsistencyDemo().evaluate()
    report = build_report("self_consistency_demo", "utilization.self_consistency_demo", {"gain": result["gain"]}, result, ["Toy self-consistency majority-vote gain demo."])
    write_report(report, output_dir / "self_consistency_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
