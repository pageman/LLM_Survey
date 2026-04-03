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
        },
        artifacts=result,
        notes=["Lite constitutional AI demo with critique-and-revision safety improvement."],
    )
    write_report(report, output_dir / "constitutional_ai_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
