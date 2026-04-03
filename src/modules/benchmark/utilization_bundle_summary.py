"""Dedicated utilization-bundle summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UtilizationBundleSummary:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        reports_dir = Path(self.reports_dir)
        filenames = [
            "retrieval_demo.json",
            "rag_demo.json",
            "react_demo.json",
            "toolformer_style_demo.json",
            "program_aided_reasoning_demo.json",
        ]
        rows = {}
        for filename in filenames:
            payload = json.loads((reports_dir / filename).read_text())
            metrics = payload["metrics"]
            score = sum(float(value) for value in metrics.values() if isinstance(value, (int, float)))
            rows[filename.replace("_demo.json", "")] = round(score, 4)
        best_name = max(rows, key=rows.get)
        return {
            "num_reports": len(filenames),
            "best_utilization_demo": best_name,
            "best_utilization_score": rows[best_name],
            "bundle_rows": rows,
        }
