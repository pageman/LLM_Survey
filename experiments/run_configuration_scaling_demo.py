"""Local configuration-scaling architecture demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.architecture import ConfigurationScalingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ConfigurationScalingDemo().evaluate()
    report = build_report(
        experiment_id="configuration_scaling_demo",
        module="architecture.configuration_scaling_demo",
        metrics={"scaling_slope": result["scaling_slope"], "max_score": result["max_score"]},
        artifacts=result,
        notes=["Lite architecture scaling demo over parameter, depth, and width configurations."],
    )
    write_report(report, output_dir / "configuration_scaling_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
