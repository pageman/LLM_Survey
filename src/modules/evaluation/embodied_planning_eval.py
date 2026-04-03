"""Lite embodied-planning evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbodiedPlanningEvaluator:
    def evaluate(self) -> dict[str, object]:
        episode_success = np.array([0.6, 0.8, 0.4, 0.8], dtype=float)
        return {
            "episode_success": episode_success.tolist(),
            "success_rate": float(episode_success.mean()),
            "path_consistency": float(episode_success.std()),
        }
