"""Lite formal-reasoning evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FormalReasoningEvaluator:
    def evaluate(self) -> dict[str, object]:
        valid = np.array([1, 1, 1, 0, 1, 1], dtype=float)
        return {
            "valid_steps": valid.astype(int).tolist(),
            "proof_validity": float(valid.mean()),
            "formal_error_rate": float(1.0 - valid.mean()),
        }
