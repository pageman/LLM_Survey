"""Lite repeated-data scaling demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepeatedDataScalingDemo:
    def evaluate(self) -> dict[str, object]:
        repeat_ratio = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        validation_loss = np.array([1.24, 1.11, 1.03, 1.09, 1.26], dtype=float)
        best_idx = int(np.argmin(validation_loss))
        return {
            "repeat_ratio": repeat_ratio.tolist(),
            "validation_loss": validation_loss.tolist(),
            "best_repeat_ratio": float(repeat_ratio[best_idx]),
            "best_validation_loss": float(validation_loss[best_idx]),
            "overfit_gap": float(validation_loss[-1] - validation_loss[0]),
        }
