"""Compare flash-attention-lite against dense attention."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import FlashAttentionComparisonDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = FlashAttentionComparisonDemo().evaluate()
    report = build_report(
        experiment_id="flash_attention_comparison_demo",
        module="systems.flash_attention_comparison_demo",
        metrics={
            "max_abs_error": result["max_abs_error"],
            "mean_abs_error": result["mean_abs_error"],
            "memory_ratio_per_block": result["memory_ratio_per_block"],
            "num_blocks": result["num_blocks"],
        },
        artifacts=result,
        notes=["Compares tiled flash-attention-lite against dense attention on the same causal-mask input."],
    )
    write_report(report, generated / "flash_attention_comparison_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
