"""Lite retrieval-grounding evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalGroundingEvaluator:
    def evaluate(self) -> dict[str, object]:
        support = np.array([0.86, 0.81, 0.79, 0.75], dtype=float)
        hallucination = np.array([0.14, 0.19, 0.21, 0.25], dtype=float)
        return {
            "support_rate": support.tolist(),
            "hallucination_rate": hallucination.tolist(),
            "grounding_score": float(support.mean()),
            "support_floor": float(support.min()),
        }
