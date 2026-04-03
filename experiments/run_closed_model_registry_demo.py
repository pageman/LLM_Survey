"""Local closed-model registry demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import ClosedModelRegistry


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ClosedModelRegistry().evaluate()
    report = build_report(
        experiment_id="closed_model_registry_demo",
        module="resources.closed_model_registry",
        metrics={
            "registry_size": result["registry_size"],
            "tooling_rate": result["tooling_rate"],
            "multimodal_rate": result["multimodal_rate"],
        },
        artifacts=result,
        notes=["Dedicated closed-model registry emphasizing API-only frontier systems."],
    )
    write_report(report, generated / "closed_model_registry_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
