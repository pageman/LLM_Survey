"""Local library-stack matrix demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import LibraryStackMatrix


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = LibraryStackMatrix().evaluate()
    report = build_report(
        experiment_id="library_stack_matrix_demo",
        module="resources.library_stack_matrix",
        metrics={
            "num_libraries": result["num_libraries"],
            "capability_coverage": result["capability_coverage"],
        },
        artifacts=result,
        notes=["Dedicated library-stack matrix covering math, autograd, serving, and tokenization roles."],
    )
    write_report(report, generated / "library_stack_matrix_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
