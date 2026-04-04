"""Local contamination demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import ContaminationExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ContaminationExperiment(seed=2).evaluate()
    report = build_report(
        experiment_id="contamination_demo",
        module="pretraining.contamination_demo",
        metrics={"max_inflation": result["max_inflation"]},
        artifacts=result,
        notes=["Contamination demo with leakage-type rows and reported-vs-true score inflation."],
    )
    write_report(report, output_dir / "contamination_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
