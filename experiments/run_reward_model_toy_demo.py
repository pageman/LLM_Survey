from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import RewardModelToy


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RewardModelToy().evaluate()
    report = build_report("reward_model_toy_demo", "adaptation.reward_model_toy", {"margin": result["margin"]}, result, ["Toy chosen-vs-rejected reward model demo."])
    write_report(report, output_dir / "reward_model_toy_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
