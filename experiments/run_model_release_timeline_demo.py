"""Local model-release timeline demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import ModelReleaseTimeline


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ModelReleaseTimeline().evaluate()
    report = build_report(
        experiment_id="model_release_timeline_demo",
        module="resources.model_release_timeline",
        metrics={
            "release_count": result["release_count"],
            "capability_gain": result["capability_gain"],
            "average_release_gap_years": result["average_release_gap_years"],
        },
        artifacts=result,
        notes=["Dedicated timeline over model-family releases and capability bands."],
    )
    write_report(report, generated / "model_release_timeline_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
