"""Lite inference-batching demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InferenceBatchingDemo:
    def evaluate(self) -> dict[str, object]:
        batch_size = np.array([1, 2, 4, 8, 16], dtype=float)
        latency = np.array([1.0, 1.28, 1.9, 3.05, 5.1], dtype=float)
        throughput = batch_size / latency
        return {
            "batch_size": batch_size.astype(int).tolist(),
            "latency": latency.tolist(),
            "throughput": throughput.round(4).tolist(),
            "max_throughput": float(throughput.max()),
            "latency_amortization": float(throughput[-1] / throughput[0]),
        }
