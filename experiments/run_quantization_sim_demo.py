"""Run the pure-NumPy quantization simulation demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import QuantizationSimDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = QuantizationSimDemo().evaluate()
    report = build_report(
        experiment_id="quantization_sim_demo",
        module="systems.quantization_sim_demo",
        metrics={
            "int8_mae": result["int8_mae"],
            "fp8_mae": result["fp8_mae"],
            "int8_compression_ratio": result["int8_compression_ratio"],
        },
        artifacts=result,
        notes=["Simulates int8 and fp8-style weight quantization in pure NumPy for educational comparison."],
    )
    write_report(report, generated / "quantization_sim_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
