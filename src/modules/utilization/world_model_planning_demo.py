"""Lite world-model planning demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorldModelPlanningDemo:
    def evaluate(self) -> dict[str, object]:
        state_values = np.array([0.21, 0.47, 0.69, 1.0], dtype=float)
        return {
            "state_values": state_values.tolist(),
            "plan_success": float(state_values[-1]),
            "state_value_gain": float(state_values[-1] - state_values[0]),
        }
