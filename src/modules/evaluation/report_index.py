"""Index generated reports into one summary artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportIndex:
    """Build a small index over generated local experiment reports."""

    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)

    def build(self) -> dict[str, Any]:
        report_paths = sorted(self.reports_dir.glob("*.json"))
        raw_reports = []
        grouped_by_module: dict[str, list[dict[str, Any]]] = {}

        for path in report_paths:
            data = json.loads(path.read_text())
            row = {
                "experiment_id": data["experiment_id"],
                "module": data["module"],
                "status": data["status"],
                "path": str(path),
            }
            raw_reports.append(row)
            grouped_by_module.setdefault(data["module"], []).append(row)

        reports = []
        modules: dict[str, list[str]] = {}
        stale_reports = []

        for module, rows in grouped_by_module.items():
            rows = sorted(rows, key=self._canonical_sort_key)
            canonical = rows[0]
            reports.append(canonical)
            modules[module] = [canonical["experiment_id"]]
            stale_reports.extend(rows[1:])

        return {
            "raw_num_reports": len(raw_reports),
            "num_reports": len(reports),
            "reports": reports,
            "modules": modules,
            "stale_report_count": len(stale_reports),
            "stale_reports": stale_reports,
        }

    @staticmethod
    def _canonical_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
        experiment_id = row["experiment_id"]
        preferred = experiment_id.endswith("_demo") or experiment_id in {"all_local_demos"}
        return (0 if preferred else 1, len(experiment_id), experiment_id, row["path"])
