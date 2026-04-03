"""Local prompt-order sensitivity demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import PromptOrderSensitivityDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PromptOrderSensitivityDemo().evaluate()
    report = build_report(
        experiment_id="prompt_order_sensitivity_demo",
        module="utilization.prompt_order_sensitivity_demo",
        metrics={"order_variance": result["order_variance"], "best_order_score": result["best_order_score"]},
        artifacts=result,
        notes=["Lite prompt-order sensitivity demo over demonstration permutations."],
    )
    write_report(report, output_dir / "prompt_order_sensitivity_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
