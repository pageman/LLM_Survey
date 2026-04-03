"""Lite code-eval style pass@k demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeEvalDemo:
    def evaluate(self) -> dict[str, object]:
        pass_at_k = np.array([0.31, 0.52, 0.68], dtype=float)
        return {
            "k_values": [1, 5, 10],
            "pass_at_k": pass_at_k.tolist(),
            "pass_at_1": float(pass_at_k[0]),
            "pass_at_10": float(pass_at_k[-1]),
        }
