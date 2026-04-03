"""Local public-model registry demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import PublicModelRegistry


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = PublicModelRegistry().evaluate()
    report = build_report(
        experiment_id="public_model_registry_demo",
        module="resources.public_model_registry",
        metrics={
            "registry_size": result["registry_size"],
            "open_license_rate": result["open_license_rate"],
            "average_context_window": result["average_context_window"],
        },
        artifacts=result,
        notes=["Dedicated public-model registry for open, inspectable model families."],
    )
    write_report(report, generated / "public_model_registry_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
