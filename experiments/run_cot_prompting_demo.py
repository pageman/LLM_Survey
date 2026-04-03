from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import CoTPromptingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CoTPromptingDemo().evaluate()
    report = build_report("cot_prompting_demo", "utilization.cot_prompting", {"gain": result["gain"]}, result, ["Toy chain-of-thought prompting gain demo."])
    write_report(report, output_dir / "cot_prompting_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
