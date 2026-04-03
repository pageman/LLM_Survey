"""Lite batch-scaling demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatchScalingDemo:
    def evaluate(self) -> dict[str, object]:
        batch_size = np.array([16, 32, 64, 128, 256], dtype=float)
        throughput = np.array([1.0, 1.7, 2.9, 4.6, 6.5], dtype=float)
        quality = np.array([1.14, 1.07, 1.0, 1.02, 1.09], dtype=float)
        best_idx = int(np.argmin(quality))
        return {
            "batch_size": batch_size.astype(int).tolist(),
            "throughput": throughput.tolist(),
            "quality": quality.tolist(),
            "best_batch_size": int(batch_size[best_idx]),
            "throughput_gain": float(throughput[-1] - throughput[0]),
        }
