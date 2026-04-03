"""Lite memory-partitioning demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryPartitioningDemo:
    def evaluate(self) -> dict[str, object]:
        shards = np.array([1, 2, 4, 8], dtype=float)
        memory_fraction = 1.0 / np.sqrt(shards)
        communication = np.log2(shards + 1.0) / 4.0
        return {
            "shards": shards.astype(int).tolist(),
            "memory_fraction": memory_fraction.round(4).tolist(),
            "communication_cost": communication.round(4).tolist(),
            "memory_saving": float(1.0 - memory_fraction[-1]),
            "max_communication_cost": float(communication[-1]),
        }
