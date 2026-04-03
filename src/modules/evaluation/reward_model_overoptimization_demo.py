"""Lite reward-model overoptimization demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardModelOveroptimizationDemo:
    def evaluate(self) -> dict[str, object]:
        reward = np.array([0.52, 0.64, 0.76], dtype=float)
        factuality = np.array([0.7, 0.63, 0.56], dtype=float)
        return {
            "reward": reward.tolist(),
            "factuality": factuality.tolist(),
            "reward_factuality_correlation": float(np.corrcoef(reward, factuality)[0, 1]),
            "overoptimization_gap": float((reward - factuality).mean()),
        }
