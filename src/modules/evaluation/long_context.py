"""Toy long-context evaluation inspired by Lost in the Middle."""

from __future__ import annotations

import numpy as np


class LongContextEvaluator:
    """Simulate retrieval/answering quality as evidence moves across context."""

    def __init__(self, context_length: int = 16, edge_boost: float = 0.35, middle_penalty: float = 0.30):
        if context_length < 3:
            raise ValueError("context_length must be at least 3")
        self.context_length = context_length
        self.edge_boost = edge_boost
        self.middle_penalty = middle_penalty

    def position_score(self, position: int) -> float:
        normalized = position / max(self.context_length - 1, 1)
        edge_distance = min(normalized, 1.0 - normalized)
        # U-shaped curve: strong at edges, weaker in the middle.
        score = 1.0 - self.middle_penalty - (2.0 * edge_distance) * (1.0 - self.edge_boost)
        if position == 0 or position == self.context_length - 1:
            score = 1.0
        return float(np.clip(score, 0.0, 1.0))

    def evaluate(self) -> dict[str, object]:
        positions = list(range(self.context_length))
        scores = [self.position_score(position) for position in positions]
        middle_index = self.context_length // 2
        return {
            "positions": positions,
            "scores": scores,
            "best_edge_score": max(scores[0], scores[-1]),
            "middle_score": scores[middle_index],
            "edge_gap": max(scores[0], scores[-1]) - scores[middle_index],
        }
