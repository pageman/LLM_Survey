"""Lite least-to-most reasoning demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LeastToMostDemo:
    def evaluate(self) -> dict[str, object]:
        direct = np.array([0.43, 0.46, 0.4], dtype=float)
        decomposed = np.array([0.64, 0.68, 0.62], dtype=float)
        return {
            "direct": direct.tolist(),
            "decomposed": decomposed.tolist(),
            "decomposition_gain": float((decomposed - direct).mean()),
            "stepwise_success": float(decomposed.mean()),
        }
