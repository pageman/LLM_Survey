"""Lite NLP-as-code structuring demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NLPAsCodeDemo:
    def evaluate(self) -> dict[str, object]:
        description_lengths = np.array([18, 23, 21], dtype=float)
        code_lengths = np.array([9, 11, 10], dtype=float)
        ratio = code_lengths / description_lengths
        return {
            "description_lengths": description_lengths.astype(int).tolist(),
            "code_lengths": code_lengths.astype(int).tolist(),
            "compression_ratio": float(ratio.mean()),
            "structuring_gain": float(1.0 - ratio.mean()),
        }
