"""Toy safety evaluation for refusal and harmlessness behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyEvaluator:
    def evaluate(self) -> dict[str, object]:
        refusal_rate = 0.88
        harmless_response_rate = 0.91
        jailbreak_success_rate = 0.12
        return {
            "refusal_rate": refusal_rate,
            "harmless_response_rate": harmless_response_rate,
            "jailbreak_success_rate": jailbreak_success_rate,
        }
