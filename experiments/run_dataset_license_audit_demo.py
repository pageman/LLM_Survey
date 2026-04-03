"""Local dataset-license audit demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.resources import DatasetLicenseAudit


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = DatasetLicenseAudit().evaluate()
    report = build_report(
        experiment_id="dataset_license_audit_demo",
        module="resources.dataset_license_audit",
        metrics={
            "dataset_count": result["dataset_count"],
            "redistributable_rate": result["redistributable_rate"],
            "mixed_license_rate": result["mixed_license_rate"],
        },
        artifacts=result,
        notes=["Dedicated audit over redistributable and mixed-license dataset slices."],
    )
    write_report(report, generated / "dataset_license_audit_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
