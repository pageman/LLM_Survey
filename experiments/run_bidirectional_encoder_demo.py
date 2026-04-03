"""Local bidirectional-encoder architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import BidirectionalEncoderDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = BidirectionalEncoderDemo().evaluate()
    report = build_report(
        experiment_id="bidirectional_encoder_demo",
        module="architecture.bidirectional_encoder_demo",
        metrics={
            "context_gain": result["context_gain"],
            "cloze_accuracy": result["cloze_accuracy"],
        },
        artifacts=result,
        notes=["Lite bidirectional encoder demo comparing bidirectional and causal context access."],
    )
    write_report(report, output_dir / "bidirectional_encoder_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
