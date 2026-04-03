"""Lite toxicity-filter demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToxicityFilterDemo:
    def evaluate(self) -> dict[str, object]:
        toxicity = np.array([0.92, 0.74, 0.31, 0.16, 0.07, 0.03], dtype=float)
        kept = toxicity < 0.35
        return {
            "raw_toxicity": toxicity.tolist(),
            "retained_mask": kept.astype(int).tolist(),
            "retention_rate": float(kept.mean()),
            "toxicity_reduction": float(toxicity.mean() - toxicity[kept].mean()),
        }
