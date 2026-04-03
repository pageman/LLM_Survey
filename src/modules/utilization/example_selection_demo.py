"""Lite example-selection demo for in-context prompting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExampleSelectionDemo:
    def evaluate(self) -> dict[str, object]:
        similarities = np.array([0.93, 0.87, 0.74, 0.58, 0.41], dtype=float)
        return {
            "similarities": similarities.tolist(),
            "selected_indices": [0, 1, 2],
            "topk_similarity": float(similarities[:3].mean()),
            "selection_gap": float(similarities[0] - similarities[-1]),
        }
