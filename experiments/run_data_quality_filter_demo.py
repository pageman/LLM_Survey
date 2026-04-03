from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DataQualityFilterDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DataQualityFilterDemo().evaluate()
    report = build_report("data_quality_filter_demo", "pretraining.data_quality_filter_demo", {"quality_gain": result["quality_gain"]}, result, ["Toy data quality filtering and toxicity reduction demo."])
    write_report(report, output_dir / "data_quality_filter_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
