"""Local memorization-versus-generalization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.crosscutting import MemorizationGeneralizationDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MemorizationGeneralizationDemo().evaluate()
    report = build_report(
        experiment_id="memorization_vs_generalization_demo",
        module="memorization_vs_generalization_demo",
        metrics={
            "generalization_gap": result["generalization_gap"],
            "privacy_risk": result["privacy_risk"],
            "transfer_score": result["transfer_score"],
        },
        artifacts=result,
        notes=["Bucketed memorization-versus-generalization demo with novelty-aware transfer behavior."],
    )
    write_report(report, output_dir / "memorization_vs_generalization_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
