"""Run the sparse family comparison board demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import SparseFamilyBoardDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = SparseFamilyBoardDemo().evaluate()
    report = build_report(
        experiment_id="sparse_family_board_demo",
        module="systems.sparse_family_board_demo",
        metrics={
            "family_count": result["family_count"],
            "best_efficiency": result["best_efficiency"],
            "mean_gap": result["mean_gap"],
        },
        artifacts=result,
        notes=["Compares block-sparse, ring, sliding-window, and token-sparse families in one lightweight board."],
    )
    write_report(report, generated / "sparse_family_board_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
