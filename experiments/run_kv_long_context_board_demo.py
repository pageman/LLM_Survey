"""Run the KV-aware long-context family board demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import KVLongContextBoardDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = KVLongContextBoardDemo().evaluate()
    report = build_report(
        experiment_id="kv_long_context_board_demo",
        module="systems.kv_long_context_board_demo",
        metrics={
            "family_count": result["family_count"],
            "best_efficiency": result["best_efficiency"],
            "mean_gap": result["mean_gap"],
        },
        artifacts=result,
        notes=["Compares flash, sliding-window, ring, and block-sparse families through a KV-aware long-context board."],
    )
    write_report(report, generated / "kv_long_context_board_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
