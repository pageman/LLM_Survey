from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DataCurriculumDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DataCurriculumDemo().evaluate()
    report = build_report("data_curriculum_demo", "pretraining.data_curriculum_demo", {"final_gain": result["final_gain"]}, result, ["Toy curriculum-vs-shuffled training comparison."])
    write_report(report, output_dir / "data_curriculum_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
