"""Local prefix-LM architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import PrefixLMDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PrefixLMDemo().evaluate()
    report = build_report(
        experiment_id="prefix_lm_demo",
        module="architecture.prefix_lm_demo",
        metrics={
            "prefix_visibility": result["prefix_visibility"],
            "target_causal_ratio": result["target_causal_ratio"],
        },
        artifacts=result,
        notes=["Lite prefix-LM demo with explicit prefix-visible and causal target masking."],
    )
    write_report(report, output_dir / "prefix_lm_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
