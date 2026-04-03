"""Adaptation-only summary view built on top of generated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ADAPTATION_EXPERIMENTS = {
    "alignment_sft_demo",
    "finetuning_demo",
    "instruction_tuning_demo",
    "peft_lora_demo",
    "preference_tuning_demo",
}


class AdaptationSummary:
    """Build a grouped and ranked summary over adaptation reports only."""

    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)

    def load_reports(self) -> list[dict[str, Any]]:
        reports = []
        for path in sorted(self.reports_dir.glob("*.json")):
            data = json.loads(path.read_text())
            if data.get("experiment_id") in ADAPTATION_EXPERIMENTS:
                reports.append(data)
        return reports

    def build(self) -> dict[str, Any]:
        reports = self.load_reports()
        rows = []
        for report in reports:
            metrics = report.get("metrics", {})
            rows.append(
                {
                    "experiment_id": report["experiment_id"],
                    "module": report["module"],
                    "baseline_loss": metrics.get("baseline_loss"),
                    "adapted_loss": metrics.get("adapted_loss"),
                    "gain": metrics.get("gain"),
                    "trainable_fraction": metrics.get("trainable_fraction"),
                }
            )

        ranked_by_gain = sorted(
            rows,
            key=lambda row: float("-inf") if row["gain"] is None else float(row["gain"]),
            reverse=True,
        )
        ranked_by_adapted_loss = sorted(
            rows,
            key=lambda row: float("inf") if row["adapted_loss"] is None else float(row["adapted_loss"]),
        )

        return {
            "num_adaptation_reports": len(rows),
            "reports": rows,
            "ranked_by_gain": ranked_by_gain,
            "ranked_by_adapted_loss": ranked_by_adapted_loss,
        }
