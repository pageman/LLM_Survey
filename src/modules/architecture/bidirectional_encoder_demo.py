"""Lite bidirectional-encoder demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BidirectionalEncoderDemo:
    def evaluate(self) -> dict[str, object]:
        causal_scores = np.array([0.42, 0.47, 0.5], dtype=float)
        bidirectional_scores = np.array([0.68, 0.73, 0.76], dtype=float)
        return {
            "causal_scores": causal_scores.tolist(),
            "bidirectional_scores": bidirectional_scores.tolist(),
            "context_gain": float((bidirectional_scores - causal_scores).mean()),
            "cloze_accuracy": float(bidirectional_scores.mean()),
        }
