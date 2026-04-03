"""Local data-mixture toy demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DataMixtureToyExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DataMixtureToyExperiment(seed=0).evaluate()
    report = build_report(
        experiment_id="data_mixture_toy_demo",
        module="pretraining.data_mixture_toy",
        metrics={"best_score": result["best_score"]},
        artifacts=result,
        notes=["Toy corpus-mixture experiment showing balanced-domain advantages."],
    )
    write_report(report, output_dir / "data_mixture_toy_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
