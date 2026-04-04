"""Run the attention-length stability demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import AttentionLengthStabilityDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = AttentionLengthStabilityDemo().evaluate()
    report = build_report(
        experiment_id="attention_length_stability_demo",
        module="systems.attention_length_stability_demo",
        metrics={
            "max_dense_flash_gap": result["max_dense_flash_gap"],
            "max_online_dense_weight_gap": result["max_online_dense_weight_gap"],
            "best_memory_ratio": result["best_memory_ratio"],
        },
        artifacts=result,
        notes=["Compares dense, tiled, and online-softmax attention behavior across increasing sequence lengths."],
    )
    write_report(report, generated / "attention_length_stability_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
