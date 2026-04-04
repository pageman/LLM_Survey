"""Run the advanced attention suite demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import AdvancedAttentionSuiteDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = AdvancedAttentionSuiteDemo().evaluate()
    report = build_report(
        experiment_id="advanced_attention_suite_demo",
        module="systems.advanced_attention_suite_demo",
        metrics={
            "max_dense_flash_gap": result["max_dense_flash_gap"],
            "max_stable_online_gap": result["max_stable_online_gap"],
            "best_dense_to_tiled_ratio": result["best_dense_to_tiled_ratio"],
        },
        artifacts=result,
        notes=["Advanced core-linked suite validating stable/online softmax and flash_attention_lite across length and scale."],
    )
    write_report(report, generated / "advanced_attention_suite_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
