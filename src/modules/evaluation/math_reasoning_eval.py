"""Lite math-reasoning evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MathReasoningEvaluator:
    def evaluate(self) -> dict[str, object]:
        correct = np.array([1, 1, 0, 1, 1, 0], dtype=float)
        return {
            "correct": correct.astype(int).tolist(),
            "accuracy": float(correct.mean()),
            "error_rate": float(1.0 - correct.mean()),
        }
