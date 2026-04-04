"""Lite speculative decoding demo with draft-token acceptance accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class SpeculativeRow(TypedDict):
    draft_length: int
    accepted_tokens: float
    verification_cost: float
    acceptance_rate: float
    step_speedup: float


@dataclass
class SpeculativeDecodingDemo:
    def evaluate(self) -> dict[str, object]:
        draft_lengths = np.array([1, 2, 4, 6], dtype=float)
        accepted = np.array([1.0, 1.7, 3.0, 4.1], dtype=float)
        verification_cost = np.array([1.0, 1.15, 1.3, 1.48], dtype=float)
        acceptance_rate = accepted / draft_lengths
        speedup_curve = accepted / verification_cost
        rows: list[SpeculativeRow] = [
            {
                "draft_length": int(length),
                "accepted_tokens": float(acc),
                "verification_cost": float(cost),
                "acceptance_rate": float(rate),
                "step_speedup": float(speed),
            }
            for length, acc, cost, rate, speed in zip(
                draft_lengths,
                accepted,
                verification_cost,
                acceptance_rate,
                speedup_curve,
            )
        ]
        return {
            "draft_lengths": draft_lengths.astype(int).tolist(),
            "accepted_tokens": accepted.tolist(),
            "verification_cost": verification_cost.tolist(),
            "acceptance_rate": float(acceptance_rate.mean()),
            "speedup": float(speedup_curve[-1] / speedup_curve[0]),
            "step_rows": rows,
        }
