"""Local red-teaming adaptation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import RedTeamingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = RedTeamingDemo().evaluate()
    report = build_report(
        experiment_id="red_teaming_demo",
        module="adaptation.red_teaming_demo",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts=result,
        notes=["Lite red-teaming demo with adversarial attack-success reduction."],
    )
    write_report(report, output_dir / "red_teaming_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
