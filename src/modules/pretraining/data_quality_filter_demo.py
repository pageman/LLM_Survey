"""Toy data quality filtering demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataQualityFilterDemo:
    def evaluate(self) -> dict[str, object]:
        raw_quality = 0.62
        filtered_quality = 0.78
        toxic_rate_raw = 0.18
        toxic_rate_filtered = 0.08
        return {
            "raw_quality": raw_quality,
            "filtered_quality": filtered_quality,
            "quality_gain": filtered_quality - raw_quality,
            "toxic_rate_raw": toxic_rate_raw,
            "toxic_rate_filtered": toxic_rate_filtered,
        }
