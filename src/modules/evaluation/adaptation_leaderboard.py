"""Presentation-oriented leaderboard derived from the adaptation summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adaptation_summary import AdaptationSummary


class AdaptationLeaderboard:
    """Build simple top-k style views over adaptation results."""

    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)

    @staticmethod
    def _sort_desc(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: float("-inf") if row.get(key) is None else float(row[key]),
            reverse=True,
        )

    @staticmethod
    def _sort_asc(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: float("inf") if row.get(key) is None else float(row[key]),
        )

    def build(self) -> dict[str, Any]:
        summary = AdaptationSummary(self.reports_dir).build()
        rows = summary["reports"]

        efficiency_rows = []
        for row in rows:
            gain = row.get("gain")
            trainable_fraction = row.get("trainable_fraction")
            efficiency = None
            if gain is not None:
                if trainable_fraction is None:
                    efficiency = float(gain)
                else:
                    efficiency = float(gain) / max(float(trainable_fraction), 1e-8)

            enriched = dict(row)
            enriched["efficiency_score"] = efficiency
            efficiency_rows.append(enriched)

        top_by_gain = self._sort_desc(efficiency_rows, "gain")
        top_by_efficiency = self._sort_desc(efficiency_rows, "efficiency_score")
        top_by_lowest_adapted_loss = self._sort_asc(efficiency_rows, "adapted_loss")

        return {
            "num_ranked": len(efficiency_rows),
            "top_by_gain": top_by_gain,
            "top_by_efficiency": top_by_efficiency,
            "top_by_lowest_adapted_loss": top_by_lowest_adapted_loss,
        }
