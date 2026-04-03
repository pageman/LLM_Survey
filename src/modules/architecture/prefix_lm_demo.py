"""Lite prefix-LM demo with explicit prefix-visible masking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrefixLMDemo:
    prefix_length: int = 3
    total_length: int = 6

    def evaluate(self) -> dict[str, object]:
        mask = np.zeros((self.total_length, self.total_length), dtype=float)
        for row in range(self.total_length):
            for col in range(self.total_length):
                if col < self.prefix_length or col <= row:
                    mask[row, col] = 1.0
        return {
            "mask": mask.astype(int).tolist(),
            "prefix_visibility": float(mask[:, : self.prefix_length].mean()),
            "target_causal_ratio": float(mask[:, self.prefix_length :].mean()),
        }
