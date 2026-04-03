"""Lite PPO-style RLHF demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PPORLFHToy:
    def evaluate(self) -> dict[str, object]:
        reward = np.array([0.34, 0.49, 0.58, 0.62], dtype=float)
        kl = np.array([0.02, 0.05, 0.08, 0.11], dtype=float)
        baseline_loss = 1.41
        adapted_loss = baseline_loss - float((reward - 0.4 * kl).mean())
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "reward_curve": reward.tolist(),
            "kl_curve": kl.tolist(),
        }
