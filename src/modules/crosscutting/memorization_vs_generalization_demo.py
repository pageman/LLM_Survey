"""Lite memorization-versus-generalization demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemorizationGeneralizationDemo:
    def evaluate(self) -> dict[str, object]:
        train = np.array([0.98, 0.97, 0.99], dtype=float)
        test = np.array([0.73, 0.69, 0.67], dtype=float)
        return {
            "train": train.tolist(),
            "test": test.tolist(),
            "generalization_gap": float((train - test).mean()),
            "privacy_risk": float(train.mean() - test.mean()),
        }
