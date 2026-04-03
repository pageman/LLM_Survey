"""Toy contamination experiment for train-test leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContaminationExperiment:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def evaluate(self, leakage_rates: list[float] | None = None) -> dict[str, object]:
        leakage_rates = leakage_rates or [0.0, 0.05, 0.1, 0.2, 0.4]
        reported_scores = []
        true_scores = []
        inflation = []
        for rate in leakage_rates:
            true = 0.55 + float(self.rng.normal(0.0, 0.01))
            leaked = true + 0.8 * rate + float(self.rng.normal(0.0, 0.01))
            reported_scores.append(min(leaked, 1.0))
            true_scores.append(true)
            inflation.append(reported_scores[-1] - true)
        return {
            "leakage_rates": leakage_rates,
            "reported_scores": reported_scores,
            "true_scores": true_scores,
            "inflation": inflation,
            "max_inflation": max(inflation),
        }
