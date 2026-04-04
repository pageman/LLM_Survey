"""Deduplication experiment with exact, near, and semantic duplicate modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeduplicationExperiment:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def evaluate(self, duplicate_rates: list[float] | None = None) -> dict[str, object]:
        duplicate_rates = duplicate_rates or [0.0, 0.1, 0.25, 0.5, 0.75]
        quality_scores = []
        privacy_risks = []
        duplicate_modes = []
        for rate in duplicate_rates:
            quality = 1.0 - 0.6 * rate + float(self.rng.normal(0.0, 0.01))
            risk = 0.1 + 0.9 * rate + float(self.rng.normal(0.0, 0.01))
            quality_scores.append(max(quality, 0.0))
            privacy_risks.append(min(max(risk, 0.0), 1.0))
            duplicate_modes.append(
                {
                    "duplicate_rate": rate,
                    "exact_duplicates": int(1000 * rate * 0.45),
                    "near_duplicates": int(1000 * rate * 0.35),
                    "semantic_duplicates": int(1000 * rate * 0.20),
                    "quality_score": quality_scores[-1],
                    "privacy_risk": privacy_risks[-1],
                }
            )
        return {
            "duplicate_rates": duplicate_rates,
            "quality_scores": quality_scores,
            "privacy_risks": privacy_risks,
            "best_quality_rate": duplicate_rates[int(np.argmax(quality_scores))],
            "duplicate_modes": duplicate_modes,
        }
