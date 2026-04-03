"""Local data-age freshness demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import DataAgeDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = DataAgeDemo().evaluate()
    report = build_report(
        experiment_id="data_age_demo",
        module="pretraining.data_age_demo",
        metrics={
            "recency_sensitivity": result["recency_sensitivity"],
            "freshness_gain": result["freshness_gain"],
        },
        artifacts=result,
        notes=["Lite data-age demo with freshness-sensitive utility curves."],
    )
    write_report(report, output_dir / "data_age_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
