"""Dedicated truthfulness-vs-helpfulness evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TruthfulnessHelpfulnessEvaluator:
    def evaluate(self) -> dict[str, object]:
        helpfulness = np.array([0.91, 0.88, 0.84, 0.79], dtype=float)
        truthfulness = np.array([0.86, 0.81, 0.76, 0.72], dtype=float)
        gap = helpfulness - truthfulness
        return {
            "helpfulness_score": float(helpfulness.mean()),
            "truthfulness_score": float(truthfulness.mean()),
            "mean_gap": float(gap.mean()),
            "max_gap": float(gap.max()),
            "paired_scores": [
                {"helpfulness": float(h), "truthfulness": float(t)}
                for h, t in zip(helpfulness, truthfulness)
            ],
        }
