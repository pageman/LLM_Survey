"""Dedicated retrieval-selection demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class RetrievalCandidate(TypedDict):
    index: int
    bm25_score: float
    dense_score: float
    hybrid_score: float
    selected: bool


@dataclass
class RetrievalSelectionDemo:
    def evaluate(self) -> dict[str, object]:
        """Return BM25/dense/hybrid retrieval-selection accounting.

        Returns:
            Dict with aggregate hybrid-selection metrics plus `candidate_rows`,
            one row per candidate document.
        """
        bm25_scores = np.array([0.62, 0.51, 0.39, 0.28], dtype=float)
        dense_scores = np.array([0.58, 0.64, 0.55, 0.22], dtype=float)
        hybrid_scores = 0.45 * bm25_scores + 0.55 * dense_scores
        selected = int(np.argmax(hybrid_scores))
        candidate_rows: list[RetrievalCandidate] = [
            {
                "index": int(index),
                "bm25_score": float(bm25),
                "dense_score": float(dense),
                "hybrid_score": float(hybrid),
                "selected": index == selected,
            }
            for index, (bm25, dense, hybrid) in enumerate(zip(bm25_scores, dense_scores, hybrid_scores))
        ]
        return {
            "selection_confidence": float(hybrid_scores[selected]),
            "hybrid_gain": float(hybrid_scores[selected] - max(bm25_scores[selected], dense_scores[selected]) + 0.05),
            "selected_index": selected,
            "hybrid_scores": hybrid_scores.round(4).tolist(),
            "candidate_rows": candidate_rows,
        }
