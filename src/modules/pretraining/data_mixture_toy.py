"""Toy data-mixture experiment for pre-training corpus composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataMixtureToyExperiment:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def evaluate(self, ratios: list[float] | None = None) -> dict[str, object]:
        ratios = ratios or [0.0, 0.25, 0.5, 0.75, 1.0]
        scores = []
        for ratio in ratios:
            # Best score near balanced domain mixing.
            score = 1.0 - abs(ratio - 0.5) * 1.2 + float(self.rng.normal(0.0, 0.02))
            scores.append(max(score, 0.0))
        best_idx = int(np.argmax(scores))
        return {
            "ratios": ratios,
            "scores": scores,
            "best_ratio": ratios[best_idx],
            "best_score": scores[best_idx],
        }
