"""Dedicated constitution-sweep demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConstitutionSweepDemo:
    def evaluate(self) -> dict[str, object]:
        principles = ["non-maleficence", "truthfulness", "privacy", "fairness"]
        gains = np.array([0.32, 0.41, 0.28, 0.35], dtype=float)
        safety = np.array([0.81, 0.84, 0.88, 0.79], dtype=float)
        best_index = int(np.argmax(gains))
        return {
            "principle_count": len(principles),
            "best_principle": principles[best_index],
            "best_gain": float(gains[best_index]),
            "mean_safety_score": float(safety.mean()),
            "rows": [
                {"principle": principle, "gain": float(gain), "safety_score": float(score)}
                for principle, gain, score in zip(principles, gains, safety)
            ],
        }
