"""Lite safety-versus-reasoning tradeoff demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyReasoningTradeoffDemo:
    def evaluate(self) -> dict[str, object]:
        capability = np.array([0.84, 0.78, 0.72], dtype=float)
        safety = np.array([0.62, 0.73, 0.85], dtype=float)
        return {
            "capability": capability.tolist(),
            "safety": safety.tolist(),
            "risk_score": float(1.0 - safety.mean()),
            "tradeoff_correlation": float(np.corrcoef(capability, safety)[0, 1]),
        }
