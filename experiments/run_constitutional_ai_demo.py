"""Local constitutional AI adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import ConstitutionalAIDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ConstitutionalAIDemo().evaluate()
    report = build_report(
        experiment_id="constitutional_ai_demo",
        module="adaptation.constitutional_ai_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
            "mean_harmfulness_before": result["mean_harmfulness_before"],
            "mean_harmfulness_after": result["mean_harmfulness_after"],
            "critique_coverage": result["critique_coverage"],
        },
        artifacts=result,
        notes=["Constitutional AI demo with explicit principles, critique pass, revision pass, and safety/helpfulness traces."],
    )
    write_report(report, output_dir / "constitutional_ai_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
