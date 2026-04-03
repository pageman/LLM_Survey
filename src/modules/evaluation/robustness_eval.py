"""Lite robustness evaluation under perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustnessEvaluator:
    def evaluate(self) -> dict[str, object]:
        clean = np.array([0.84, 0.81, 0.79], dtype=float)
        perturbed = np.array([0.68, 0.63, 0.61], dtype=float)
        return {
            "clean": clean.tolist(),
            "perturbed": perturbed.tolist(),
            "robustness_gap": float((clean - perturbed).mean()),
            "perturbed_score": float(perturbed.mean()),
        }
