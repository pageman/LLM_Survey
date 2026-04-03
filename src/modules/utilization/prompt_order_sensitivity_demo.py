"""Lite prompt-order sensitivity demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PromptOrderSensitivityDemo:
    def evaluate(self) -> dict[str, object]:
        permutation_scores = np.array([0.72, 0.81, 0.69, 0.77], dtype=float)
        return {
            "permutation_scores": permutation_scores.tolist(),
            "order_variance": float(permutation_scores.var()),
            "best_order_score": float(permutation_scores.max()),
        }
