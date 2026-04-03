"""Dedicated context-packing demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContextPackingDemo:
    def evaluate(self) -> dict[str, object]:
        examples = np.array([180, 220, 140, 260, 120], dtype=float)
        max_window = 512.0
        packed_tokens = examples[0] + examples[2] + examples[4]
        naive_padding = max_window * len(examples)
        packed_efficiency = packed_tokens / max_window
        packing_gain = (naive_padding - (2 * max_window)) / naive_padding
        return {
            "packed_efficiency": float(packed_efficiency),
            "packing_gain": float(packing_gain),
            "packed_examples": [0, 2, 4],
            "token_lengths": examples.astype(int).tolist(),
        }
