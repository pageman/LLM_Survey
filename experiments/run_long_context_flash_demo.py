"""Run the long-context flash-attention comparison demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import LongContextFlashDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = LongContextFlashDemo().evaluate()
    report = build_report(
        experiment_id="long_context_flash_demo",
        module="systems.long_context_flash_demo",
        metrics={
            "mean_abs_error": result["mean_abs_error"],
            "max_abs_error": result["max_abs_error"],
            "dense_to_tiled_ratio": result["dense_to_tiled_ratio"],
        },
        artifacts=result,
        notes=["Long-context dense-vs-flash comparison using the tiled flash_attention_lite helper."],
    )
    write_report(report, generated / "long_context_flash_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
