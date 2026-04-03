"""Lite architecture configuration-scaling demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigurationScalingDemo:
    def evaluate(self) -> dict[str, object]:
        params = np.array([25, 80, 250, 800], dtype=float)
        depth = np.array([6, 12, 24, 36], dtype=float)
        width = np.array([256, 384, 512, 768], dtype=float)
        score = np.array([0.48, 0.57, 0.66, 0.74], dtype=float)
        slope = np.polyfit(np.log10(params), score, 1)[0]
        return {
            "params_millions": params.tolist(),
            "depth": depth.astype(int).tolist(),
            "width": width.astype(int).tolist(),
            "score": score.tolist(),
            "scaling_slope": float(slope),
            "max_score": float(score.max()),
        }
