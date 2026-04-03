"""Position bias summary metrics built on long-context evaluation."""

from __future__ import annotations

import numpy as np

from .long_context import LongContextEvaluator


class PositionBiasEvaluator:
    """Summarize edge preference and middle degradation."""

    def __init__(self, context_length: int = 16):
        self.long_context = LongContextEvaluator(context_length=context_length)

    def evaluate(self) -> dict[str, object]:
        result = self.long_context.evaluate()
        scores = np.array(result["scores"], dtype=float)
        n = len(scores)
        first_third = scores[: max(1, n // 3)]
        middle_third = scores[n // 3 : max(n // 3 + 1, 2 * n // 3)]
        last_third = scores[max(2 * n // 3, 1) :]

        edge_mean = float(np.mean(np.concatenate([first_third, last_third])))
        middle_mean = float(np.mean(middle_third))
        return {
            "positions": result["positions"],
            "scores": result["scores"],
            "edge_mean": edge_mean,
            "middle_mean": middle_mean,
            "edge_over_middle_ratio": float(edge_mean / max(middle_mean, 1e-8)),
            "max_score_position": int(np.argmax(scores)),
            "min_score_position": int(np.argmin(scores)),
        }
