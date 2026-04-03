"""Lite constitutional AI demo with critique-and-revision scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConstitutionalAIDemo:
    def evaluate(self) -> dict[str, object]:
        harmful_scores = np.array([0.64, 0.56, 0.49], dtype=float)
        revised_scores = np.array([0.28, 0.24, 0.21], dtype=float)
        baseline_loss = 1.29
        adapted_loss = baseline_loss - float((harmful_scores.mean() - revised_scores.mean()) * 0.8)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "harmful_scores": harmful_scores.tolist(),
            "revised_scores": revised_scores.tolist(),
        }
