"""Local toxicity-filter pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import ToxicityFilterDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ToxicityFilterDemo().evaluate()
    report = build_report(
        experiment_id="toxicity_filter_demo",
        module="pretraining.toxicity_filter_demo",
        metrics={"retention_rate": result["retention_rate"], "toxicity_reduction": result["toxicity_reduction"]},
        artifacts=result,
        notes=["Toxicity filtering demo with threshold sweep and retention-quality tradeoffs."],
    )
    write_report(report, output_dir / "toxicity_filter_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
