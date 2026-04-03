"""Dedicated KV-cache fragmentation demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KVCacheFragmentationDemo:
    def evaluate(self) -> dict[str, object]:
        sequence_buckets = np.array([128, 256, 512, 1024], dtype=float)
        ideal_utilization = np.array([0.96, 0.93, 0.9, 0.88], dtype=float)
        fragmented_utilization = np.array([0.84, 0.77, 0.69, 0.58], dtype=float)
        fragmentation_penalty = ideal_utilization - fragmented_utilization
        return {
            "bucket_count": int(sequence_buckets.size),
            "mean_fragmentation_penalty": float(fragmentation_penalty.mean()),
            "worst_case_penalty": float(fragmentation_penalty.max()),
            "bucket_lengths": sequence_buckets.astype(int).tolist(),
            "ideal_utilization": ideal_utilization.tolist(),
            "fragmented_utilization": fragmented_utilization.tolist(),
        }
