"""Lite preference-data quality demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreferenceDataQualityDemo:
    def evaluate(self) -> dict[str, object]:
        preference_quality = np.array([0.87, 0.69, 0.51, 0.9], dtype=float)
        baseline_loss = 1.28
        adapted_loss = baseline_loss - float((preference_quality.mean() - 0.5) * 0.6)
        return {
            "preference_quality": preference_quality.tolist(),
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
        }
