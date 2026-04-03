"""Local example-selection utilization demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import ExampleSelectionDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ExampleSelectionDemo().evaluate()
    report = build_report(
        experiment_id="example_selection_demo",
        module="utilization.example_selection_demo",
        metrics={
            "topk_similarity": result["topk_similarity"],
            "selection_gap": result["selection_gap"],
        },
        artifacts=result,
        notes=["Lite example-selection demo for in-context retrieval of demonstrations."],
    )
    write_report(report, output_dir / "example_selection_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
