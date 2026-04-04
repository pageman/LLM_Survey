"""Run the scale-sensitive stability demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import ScaleStabilityDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ScaleStabilityDemo().evaluate()
    report = build_report(
        experiment_id="scale_stability_demo",
        module="systems.scale_stability_demo",
        metrics={
            "worst_naive_finite_fraction": result["worst_naive_finite_fraction"],
            "best_stable_finite_fraction": result["best_stable_finite_fraction"],
            "max_stable_online_gap": result["max_stable_online_gap"],
        },
        artifacts=result,
        notes=["Sweeps input scales to compare naive, stable, online, and Welford-style normalization behavior."],
    )
    write_report(report, generated / "scale_stability_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
