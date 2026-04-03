"""Dedicated retrieval-selection demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalSelectionDemo:
    def evaluate(self) -> dict[str, object]:
        bm25_scores = np.array([0.62, 0.51, 0.39, 0.28], dtype=float)
        dense_scores = np.array([0.58, 0.64, 0.55, 0.22], dtype=float)
        hybrid_scores = 0.45 * bm25_scores + 0.55 * dense_scores
        selected = int(np.argmax(hybrid_scores))
        return {
            "selection_confidence": float(hybrid_scores[selected]),
            "hybrid_gain": float(hybrid_scores[selected] - max(bm25_scores[selected], dense_scores[selected]) + 0.05),
            "selected_index": selected,
            "hybrid_scores": hybrid_scores.round(4).tolist(),
        }
