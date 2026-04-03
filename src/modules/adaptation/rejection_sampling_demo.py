"""Lite rejection-sampling adaptation demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RejectionSamplingDemo:
    def evaluate(self) -> dict[str, object]:
        candidate_rewards = np.array([0.41, 0.74, 0.66, 0.58], dtype=float)
        baseline_loss = 1.35
        adapted_loss = baseline_loss - float(candidate_rewards.max() * 0.3)
        return {
            "candidate_rewards": candidate_rewards.tolist(),
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
        }
