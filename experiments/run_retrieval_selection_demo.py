"""Local retrieval-selection demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.utilization import RetrievalSelectionDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = RetrievalSelectionDemo().evaluate()
    report = build_report(
        experiment_id="retrieval_selection_demo",
        module="utilization.retrieval_selection_demo",
        metrics={
            "selection_confidence": result["selection_confidence"],
            "hybrid_gain": result["hybrid_gain"],
        },
        artifacts=result,
        notes=["Dedicated retrieval-selection demo over dense, sparse, and hybrid scores."],
    )
    write_report(report, generated / "retrieval_selection_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
