"""Lite scratchpad reasoning demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScratchpadDemo:
    def evaluate(self) -> dict[str, object]:
        plain = np.array([0.5, 0.53, 0.48], dtype=float)
        scratchpad = np.array([0.66, 0.69, 0.64], dtype=float)
        return {
            "plain": plain.tolist(),
            "scratchpad": scratchpad.tolist(),
            "scratchpad_gain": float((scratchpad - plain).mean()),
            "trace_consistency": float(scratchpad.mean()),
        }
