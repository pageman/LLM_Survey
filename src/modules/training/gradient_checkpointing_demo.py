"""Lite gradient checkpointing tradeoff demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GradientCheckpointingDemo:
    def evaluate(self) -> dict[str, object]:
        memory_fraction = np.array([1.0, 0.72, 0.57], dtype=float)
        runtime_fraction = np.array([1.0, 1.08, 1.19], dtype=float)
        return {
            "memory_fraction": memory_fraction.tolist(),
            "runtime_fraction": runtime_fraction.tolist(),
            "memory_reduction": float(1.0 - memory_fraction[-1]),
            "recompute_overhead": float(runtime_fraction[-1] - 1.0),
        }
