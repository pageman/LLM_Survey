"""Local objective-mixture training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import ObjectiveMixtureDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ObjectiveMixtureDemo().evaluate()
    report = build_report(
        experiment_id="objective_mixture_demo",
        module="training.objective_mixture_demo",
        metrics={
            "best_mixture_loss": result["best_mixture_loss"],
            "mixture_gain": result["mixture_gain"],
        },
        artifacts=result,
        notes=["Lite objective-mixture demo with explicit weighting search across multiple training objectives."],
    )
    write_report(report, output_dir / "objective_mixture_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
