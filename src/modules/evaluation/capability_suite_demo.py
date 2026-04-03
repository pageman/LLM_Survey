"""Lite capability-suite summary demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapabilitySuiteDemo:
    def evaluate(self) -> dict[str, object]:
        subscores = np.array([0.73, 0.68, 0.64, 0.59, 0.62], dtype=float)
        return {
            "subscores": subscores.tolist(),
            "suite_average": float(subscores.mean()),
            "suite_minimum": float(subscores.min()),
        }
