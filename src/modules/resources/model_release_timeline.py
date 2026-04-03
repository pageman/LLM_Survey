"""Dedicated model-release timeline demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelReleaseTimeline:
    def evaluate(self) -> dict[str, object]:
        releases = [
            {"year": 2018, "model": "GPT", "capability_band": 0.18},
            {"year": 2020, "model": "GPT-3", "capability_band": 0.44},
            {"year": 2022, "model": "PaLM", "capability_band": 0.56},
            {"year": 2023, "model": "Llama 2", "capability_band": 0.62},
            {"year": 2024, "model": "Llama 3.1", "capability_band": 0.73},
            {"year": 2025, "model": "Open-weight reasoning family", "capability_band": 0.81},
        ]
        capability_gain = releases[-1]["capability_band"] - releases[0]["capability_band"]
        cadence_years = (releases[-1]["year"] - releases[0]["year"]) / (len(releases) - 1)
        return {
            "release_count": len(releases),
            "capability_gain": capability_gain,
            "average_release_gap_years": cadence_years,
            "releases": releases,
        }
