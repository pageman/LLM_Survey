"""Local encoder-decoder architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import EncoderDecoderDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = EncoderDecoderDemo().evaluate()
    report = build_report(
        experiment_id="encoder_decoder_demo",
        module="architecture.encoder_decoder_demo",
        metrics={
            "cross_attention_focus": result["cross_attention_focus"],
            "copy_accuracy": result["copy_accuracy"],
        },
        artifacts=result,
        notes=["Lite encoder-decoder demo with explicit encoder states, decoder queries, and cross-attention."],
    )
    write_report(report, output_dir / "encoder_decoder_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
