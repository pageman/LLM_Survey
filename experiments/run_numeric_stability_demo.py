"""Run the numeric stability comparison demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import NumericStabilityDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = NumericStabilityDemo().evaluate()
    report = build_report(
        experiment_id="numeric_stability_demo",
        module="systems.numeric_stability_demo",
        metrics={
            "stable_online_gap": result["stable_online_gap"],
            "welford_reference_gap": result["welford_reference_gap"],
            "stable_softmax_finite_fraction": result["stable_softmax_finite_fraction"],
        },
        artifacts=result,
        notes=["Compares naive, stable, online, and Welford-style normalization behavior on numerically extreme inputs."],
    )
    write_report(report, generated / "numeric_stability_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
