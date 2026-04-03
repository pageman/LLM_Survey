"""Lite data-age freshness demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataAgeDemo:
    def evaluate(self) -> dict[str, object]:
        age_years = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
        utility = np.array([0.92, 0.89, 0.83, 0.74, 0.61], dtype=float)
        return {
            "age_years": age_years.tolist(),
            "utility": utility.tolist(),
            "recency_sensitivity": float((utility[0] - utility[-1]) / age_years[-1]),
            "freshness_gain": float(utility[0] - utility[2]),
        }
