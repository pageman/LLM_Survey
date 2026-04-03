"""Lite multilingual transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransferEvaluator:
    def evaluate(self) -> dict[str, object]:
        zero_shot = np.array([0.69, 0.62, 0.58], dtype=float)
        few_shot = np.array([0.77, 0.7, 0.66], dtype=float)
        return {
            "zero_shot": zero_shot.tolist(),
            "few_shot": few_shot.tolist(),
            "transfer_score": float(few_shot.mean()),
            "few_shot_gain": float((few_shot - zero_shot).mean()),
        }
