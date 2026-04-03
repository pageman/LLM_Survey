"""Lite multilingual architecture demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultilingualArchitectureDemo:
    def evaluate(self) -> dict[str, object]:
        language_transfer = np.array([0.79, 0.74, 0.7, 0.67], dtype=float)
        return {
            "language_transfer": language_transfer.tolist(),
            "parameter_sharing_score": float(language_transfer.mean()),
            "transfer_score": float(language_transfer.min() / language_transfer.max()),
        }
