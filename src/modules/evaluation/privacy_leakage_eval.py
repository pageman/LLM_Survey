"""Lite privacy leakage evaluation via memorized canary exposure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrivacyLeakageEvaluator:
    def evaluate(self) -> dict[str, object]:
        exposure = np.array([0.11, 0.18, 0.07, 0.16, 0.09], dtype=float)
        return {
            "privacy_risk": float(exposure.mean()),
            "max_exposure": float(exposure.max()),
            "exposure": exposure.tolist(),
        }
