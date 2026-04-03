from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.applications import CodeGenerationDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CodeGenerationDemo().evaluate()
    report = build_report("code_generation_demo", "applications.code_generation_demo", {"pass_rate": result["pass_rate"]}, result, ["Toy code generation application demo."])
    write_report(report, output_dir / "code_generation_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
