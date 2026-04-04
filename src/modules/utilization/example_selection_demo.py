"""Lite example-selection demo for in-context prompting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class ExampleCandidate(TypedDict):
    index: int
    similarity: float
    diversity_penalty: float
    adjusted_score: float
    selected: bool


@dataclass
class ExampleSelectionDemo:
    def evaluate(self) -> dict[str, object]:
        similarities = np.array([0.93, 0.87, 0.74, 0.58, 0.41], dtype=float)
        diversity_penalty = np.array([0.00, 0.02, 0.06, 0.03, 0.01], dtype=float)
        adjusted = similarities - diversity_penalty
        selected = [0, 1, 2]
        candidate_rows: list[ExampleCandidate] = [
            {
                "index": int(index),
                "similarity": float(score),
                "diversity_penalty": float(penalty),
                "adjusted_score": float(adjusted_score),
                "selected": index in selected,
            }
            for index, (score, penalty, adjusted_score) in enumerate(zip(similarities, diversity_penalty, adjusted))
        ]
        return {
            "similarities": similarities.tolist(),
            "selected_indices": selected,
            "topk_similarity": float(similarities[:3].mean()),
            "selection_gap": float(similarities[0] - similarities[-1]),
            "adjusted_topk_similarity": float(adjusted[selected].mean()),
            "candidate_rows": candidate_rows,
        }
