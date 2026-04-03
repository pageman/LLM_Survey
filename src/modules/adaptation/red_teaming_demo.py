"""Lite red-teaming adaptation demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RedTeamingDemo:
    def evaluate(self) -> dict[str, object]:
        attack_success = np.array([0.56, 0.49, 0.41, 0.34], dtype=float)
        baseline_loss = 1.28
        adapted_loss = baseline_loss - float((attack_success[0] - attack_success[-1]) * 0.8)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "attack_success": attack_success.tolist(),
        }
