"""Local warmup-decay training demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.training import WarmupDecayDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = WarmupDecayDemo().evaluate()
    report = build_report(
        experiment_id="warmup_decay_demo",
        module="training.warmup_decay_demo",
        metrics={
            "peak_step": result["peak_step"],
            "stability_score": result["stability_score"],
        },
        artifacts=result,
        notes=["Lite warmup-decay schedule demo with peak-step and stability tracking."],
    )
    write_report(report, output_dir / "warmup_decay_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
