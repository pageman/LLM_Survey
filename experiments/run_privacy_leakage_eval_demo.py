"""Local privacy leakage evaluation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.evaluation import PrivacyLeakageEvaluator


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PrivacyLeakageEvaluator().evaluate()
    report = build_report(
        experiment_id="privacy_leakage_eval_demo",
        module="evaluation.privacy_leakage_eval",
        metrics={
            "privacy_risk": result["privacy_risk"],
            "max_exposure": result["max_exposure"],
        },
        artifacts=result,
        notes=["Lite privacy leakage demo with memorized-canary exposure accounting."],
    )
    write_report(report, output_dir / "privacy_leakage_eval_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
