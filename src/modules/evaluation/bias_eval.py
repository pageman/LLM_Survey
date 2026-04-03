"""Toy bias evaluation for differential response patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BiasEvaluator:
    def evaluate(self) -> dict[str, object]:
        stereotype_score = 0.24
        counterfactual_gap = 0.17
        fairness_score = 0.79
        return {
            "stereotype_score": stereotype_score,
            "counterfactual_gap": counterfactual_gap,
            "fairness_score": fairness_score,
        }
