"""Local constitution-sweep demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import ConstitutionSweepDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ConstitutionSweepDemo().evaluate()
    report = build_report(
        experiment_id="constitution_sweep_demo",
        module="adaptation.constitution_sweep_demo",
        metrics={
            "principle_count": result["principle_count"],
            "best_gain": result["best_gain"],
            "mean_safety_score": result["mean_safety_score"],
        },
        artifacts=result,
        notes=["Dedicated sweep over constitutional principle choices."],
    )
    write_report(report, generated / "constitution_sweep_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
