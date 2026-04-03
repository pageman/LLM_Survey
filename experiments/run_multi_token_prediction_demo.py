from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import MultiTokenPredictionDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MultiTokenPredictionDemo().evaluate([1, 2, 3], horizon=3)
    report = build_report("multi_token_prediction_demo", "pretraining.multi_token_prediction", {"sample_efficiency_gain": result["sample_efficiency_gain"]}, result, ["Toy multi-token prediction horizon demo."])
    write_report(report, output_dir / "multi_token_prediction_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
