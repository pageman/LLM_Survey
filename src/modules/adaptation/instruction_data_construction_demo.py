"""Lite instruction-data construction demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InstructionDataConstructionDemo:
    def evaluate(self) -> dict[str, object]:
        diversity = np.array([0.53, 0.65, 0.73], dtype=float)
        quality = np.array([0.49, 0.64, 0.75], dtype=float)
        baseline_loss = 1.39
        adapted_loss = baseline_loss - float((diversity * quality).mean() * 0.5)
        return {
            "diversity": diversity.tolist(),
            "quality": quality.tolist(),
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
        }
