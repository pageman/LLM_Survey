"""Lite long-tail behavior evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LongTailBehaviorEvaluator:
    def evaluate(self) -> dict[str, object]:
        head = np.array([0.88, 0.85, 0.83], dtype=float)
        tail = np.array([0.54, 0.49, 0.46], dtype=float)
        return {
            "head": head.tolist(),
            "tail": tail.tolist(),
            "head_tail_gap": float((head - tail).mean()),
            "tail_score": float(tail.mean()),
        }
