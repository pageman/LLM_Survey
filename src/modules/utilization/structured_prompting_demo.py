"""Lite structured prompting demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StructuredPromptingDemo:
    def evaluate(self) -> dict[str, object]:
        raw_scores = np.array([0.52, 0.57, 0.61], dtype=float)
        structured_scores = np.array([0.72, 0.77, 0.81], dtype=float)
        return {
            "raw_scores": raw_scores.tolist(),
            "structured_scores": structured_scores.tolist(),
            "schema_gain": float((structured_scores - raw_scores).mean()),
            "structured_success": float(structured_scores.mean()),
        }
