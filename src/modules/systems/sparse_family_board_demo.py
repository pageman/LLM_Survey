"""Compare sparse attention families in one lightweight board."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block_sparse_attention_demo import BlockSparseAttentionDemo
from .ring_attention_demo import RingAttentionDemo
from .sliding_window_kv_demo import SlidingWindowKVDemo
from .sparse_attention_demo import SparseAttentionDemo


@dataclass
class SparseFamilyBoardDemo:
    def evaluate(self) -> dict[str, object]:
        variants = [
            ("block_sparse", BlockSparseAttentionDemo(seed=1).evaluate()),
            ("ring", RingAttentionDemo(seed=2).evaluate()),
            ("sliding_window", SlidingWindowKVDemo(seed=3).evaluate()),
            ("token_sparse", SparseAttentionDemo(seed=4).evaluate()),
        ]
        rows = []
        for family, result in variants:
            if family == "sliding_window":
                efficiency = float(result["cache_reduction"])
            else:
                efficiency = float(result.get("sparsity", 1.0 - result.get("mask_density", result.get("visibility_density", 0.0))))
            rows.append(
                {
                    "family": family,
                    "efficiency": efficiency,
                    "approximation_gap": float(result["approximation_gap"]),
                }
            )
        efficiencies = np.array([row["efficiency"] for row in rows], dtype=float)
        gaps = np.array([row["approximation_gap"] for row in rows], dtype=float)
        return {
            "rows": rows,
            "family_count": len(rows),
            "best_efficiency": float(efficiencies.max()),
            "best_gap": float(gaps.min()),
            "mean_efficiency": float(efficiencies.mean()),
            "mean_gap": float(gaps.mean()),
        }
