"""Lite alignment-data filter demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentDataFilterDemo:
    def evaluate(self) -> dict[str, object]:
        raw_scores = np.array([0.61, 0.58, 0.53, 0.39], dtype=float)
        filtered_scores = raw_scores[:3]
        baseline_loss = 1.34
        adapted_loss = baseline_loss - float((filtered_scores.mean() - raw_scores.mean()) * 0.9)
        return {
            "raw_scores": raw_scores.tolist(),
            "filtered_scores": filtered_scores.tolist(),
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
        }
