"""Toy chain-of-thought prompting demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoTPromptingDemo:
    def evaluate(self) -> dict[str, object]:
        direct_score = 0.61
        cot_score = 0.79
        return {
            "direct_score": direct_score,
            "cot_score": cot_score,
            "gain": cot_score - direct_score,
        }
