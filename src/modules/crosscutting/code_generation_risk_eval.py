"""Lite code-generation risk evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeGenerationRiskEvaluator:
    def evaluate(self) -> dict[str, object]:
        unsafe_pattern_rate = np.array([0.22, 0.19, 0.25], dtype=float)
        return {
            "unsafe_pattern_rate": unsafe_pattern_rate.tolist(),
            "risk_score": float(unsafe_pattern_rate.mean()),
            "max_risk": float(unsafe_pattern_rate.max()),
        }
