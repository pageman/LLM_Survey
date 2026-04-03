"""Local module-provenance dashboard demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.reporting import ModuleProvenanceDashboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = ModuleProvenanceDashboard(generated).build()
    report = build_report(
        experiment_id="module_provenance_dashboard_demo",
        module="reporting.module_provenance_dashboard",
        metrics={
            "dedicated_count": result["dedicated_count"],
            "generated_count": result["generated_count"],
            "dedicated_fraction": result["dedicated_fraction"],
        },
        artifacts=result,
        notes=["Dedicated provenance dashboard splitting first-class demos from generated fallback reports."],
    )
    write_report(report, generated / "module_provenance_dashboard_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
