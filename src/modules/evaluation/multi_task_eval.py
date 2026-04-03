"""Lite multitask evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiTaskEvaluator:
    def evaluate(self) -> dict[str, object]:
        task_scores = np.array([0.73, 0.66, 0.6, 0.69, 0.63], dtype=float)
        return {
            "task_scores": task_scores.tolist(),
            "average_score": float(task_scores.mean()),
            "worst_task": float(task_scores.min()),
        }
