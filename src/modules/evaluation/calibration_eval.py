"""Toy calibration evaluation for confidence vs. correctness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationEvaluator:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def evaluate(self) -> dict[str, object]:
        confidences = np.array([0.55, 0.65, 0.75, 0.85, 0.95], dtype=float)
        accuracies = np.array([0.58, 0.64, 0.68, 0.72, 0.70], dtype=float)
        ece = float(np.mean(np.abs(confidences - accuracies)))
        return {
            "confidences": confidences.tolist(),
            "accuracies": accuracies.tolist(),
            "ece": ece,
        }
