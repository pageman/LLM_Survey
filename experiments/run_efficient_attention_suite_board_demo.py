"""Run the unified efficient attention suite board demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import EfficientAttentionSuiteBoardDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = EfficientAttentionSuiteBoardDemo().evaluate()
    report = build_report(
        experiment_id="efficient_attention_suite_board_demo",
        module="systems.efficient_attention_suite_board_demo",
        metrics={
            "family_count": result["family_count"],
            "best_efficiency": result["best_efficiency"],
            "lowest_quality_gap": result["lowest_quality_gap"],
        },
        artifacts=result,
        notes=["Unified efficient-attention board spanning flash, sliding-window, ring, block-sparse, and quantization-aware angles."],
    )
    write_report(report, generated / "efficient_attention_suite_board_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
