"""Lite out-of-distribution evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OutOfDistributionEvaluator:
    def evaluate(self) -> dict[str, object]:
        in_distribution = np.array([0.83, 0.80, 0.78], dtype=float)
        ood = np.array([0.64, 0.60, 0.57], dtype=float)
        return {
            "in_distribution": in_distribution.tolist(),
            "ood": ood.tolist(),
            "ood_gap": float((in_distribution - ood).mean()),
            "ood_score": float(ood.mean()),
        }
