"""Local speculative decoding systems demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import SpeculativeDecodingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = SpeculativeDecodingDemo().evaluate()
    report = build_report(
        experiment_id="speculative_decoding_demo",
        module="systems.speculative_decoding_demo",
        metrics={
            "acceptance_rate": result["acceptance_rate"],
            "speedup": result["speedup"],
        },
        artifacts=result,
        notes=["Lite speculative decoding demo with draft acceptance and verification cost accounting."],
    )
    write_report(report, output_dir / "speculative_decoding_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
