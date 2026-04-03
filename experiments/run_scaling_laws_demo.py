"""Local scaling laws demo.

Usage:
    python3 experiments/run_scaling_laws_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import run_default_scaling_suite


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = run_default_scaling_suite(seed=0)
    result = build_report(
        experiment_id="scaling_laws_demo",
        module="pretraining.scaling_laws",
        metrics={
            "parameter_b": suite["parameter_scaling"]["fit"]["b"],
            "data_b": suite["data_scaling"]["fit"]["b"],
            "compute_b": suite["compute_scaling"]["fit"]["b"],
        },
        artifacts=suite,
        notes=["Power-law scaling demo with local NumPy-only fitting."],
    )
    write_report(result, output_dir / "scaling_laws_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
