"""Safety-versus-reasoning tradeoff demo with trajectory-level behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyReasoningTradeoffDemo:
    def evaluate(self) -> dict[str, object]:
        settings = [
            {"setting": "capability_first", "reasoning_quality": 0.84, "safety_refusal": 0.62, "unsafe_compliance": 0.19},
            {"setting": "balanced", "reasoning_quality": 0.78, "safety_refusal": 0.73, "unsafe_compliance": 0.11},
            {"setting": "safety_first", "reasoning_quality": 0.72, "safety_refusal": 0.85, "unsafe_compliance": 0.05},
        ]
        capability = np.array([item["reasoning_quality"] for item in settings], dtype=float)
        safety = np.array([item["safety_refusal"] for item in settings], dtype=float)
        return {
            "capability": capability.tolist(),
            "safety": safety.tolist(),
            "risk_score": float(1.0 - safety.mean()),
            "tradeoff_correlation": float(np.corrcoef(capability, safety)[0, 1]),
            "reasoning_drop": float(capability[0] - capability[-1]),
            "settings": settings,
        }
