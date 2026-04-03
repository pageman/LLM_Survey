"""Local fidelity-band dashboard demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.reporting import FidelityBandDashboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = FidelityBandDashboard(generated).build()
    report = build_report(
        experiment_id="fidelity_band_dashboard_demo",
        module="reporting.fidelity_band_dashboard",
        metrics={
            "mechanism_level_count": result["mechanism_level_count"],
            "survey_map_count": result["survey_map_count"],
            "mechanism_level_fraction": result["mechanism_level_fraction"],
        },
        artifacts=result,
        notes=["Dedicated fidelity-band dashboard splitting mechanism-level demos from survey-map dashboards."],
    )
    write_report(report, generated / "fidelity_band_dashboard_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
