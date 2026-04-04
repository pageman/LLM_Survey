"""Run the flash-attention block-size sweep demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import FlashBlockSweepDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = FlashBlockSweepDemo().evaluate()
    report = build_report(
        experiment_id="flash_block_sweep_demo",
        module="systems.flash_block_sweep_demo",
        metrics={
            "best_dense_to_tiled_ratio": result["best_dense_to_tiled_ratio"],
            "best_mean_abs_error": result["best_mean_abs_error"],
        },
        artifacts=result,
        notes=["Sweeps block sizes for flash_attention_lite against dense attention to expose the error-vs-memory tradeoff."],
    )
    write_report(report, generated / "flash_block_sweep_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
