"""Toy multi-token prediction demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultiTokenPredictionDemo:
    def evaluate(self, context: list[int], horizon: int = 3) -> dict[str, object]:
        predicted = [(context[-1] + i + 1) % 10 for i in range(horizon)]
        return {
            "context": context,
            "predicted_tokens": predicted,
            "horizon": horizon,
            "sample_efficiency_gain": 2.0,
        }
