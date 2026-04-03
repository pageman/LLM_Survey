"""Lite speculative decoding demo with draft-token acceptance accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeculativeDecodingDemo:
    def evaluate(self) -> dict[str, object]:
        draft_lengths = np.array([1, 2, 4, 6], dtype=float)
        accepted = np.array([1.0, 1.7, 3.0, 4.1], dtype=float)
        verification_cost = np.array([1.0, 1.15, 1.3, 1.48], dtype=float)
        acceptance_rate = accepted / draft_lengths
        speedup_curve = accepted / verification_cost
        return {
            "draft_lengths": draft_lengths.astype(int).tolist(),
            "accepted_tokens": accepted.tolist(),
            "verification_cost": verification_cost.tolist(),
            "acceptance_rate": float(acceptance_rate.mean()),
            "speedup": float(speedup_curve[-1] / speedup_curve[0]),
        }
