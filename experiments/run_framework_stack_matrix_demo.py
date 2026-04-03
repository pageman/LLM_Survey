"""Local framework-stack matrix demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import FrameworkStackMatrix


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = FrameworkStackMatrix().evaluate()
    report = build_report(
        experiment_id="framework_stack_matrix_demo",
        module="resources.framework_stack_matrix",
        metrics={
            "num_frameworks": result["num_frameworks"],
            "serving_ready_fraction": result["serving_ready_fraction"],
            "distributed_ready_fraction": result["distributed_ready_fraction"],
        },
        artifacts=result,
        notes=["Dedicated framework-stack matrix covering training, distributed execution, and serving."],
    )
    write_report(report, generated / "framework_stack_matrix_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
