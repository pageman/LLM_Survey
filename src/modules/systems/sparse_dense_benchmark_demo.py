"""Benchmark-style comparison between dense and sparse attention approximations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sliding_window_kv_demo import SlidingWindowKVDemo
from .sparse_attention_demo import SparseAttentionDemo


@dataclass
class SparseDenseBenchmarkDemo:
    def evaluate(self) -> dict[str, object]:
        sparse_variants = [
            SparseAttentionDemo(seq_len=24, d_model=8, local_window=2, global_stride=4, seed=1).evaluate(),
            SparseAttentionDemo(seq_len=24, d_model=8, local_window=3, global_stride=6, seed=2).evaluate(),
        ]
        sliding_variants = [
            SlidingWindowKVDemo(seq_len=24, d_model=8, window_size=6, sink_tokens=2, seed=3).evaluate(),
            SlidingWindowKVDemo(seq_len=24, d_model=8, window_size=8, sink_tokens=4, seed=4).evaluate(),
        ]
        rows = []
        for index, result in enumerate(sparse_variants):
            rows.append(
                {
                    "family": "block_sparse",
                    "variant": f"sparse_{index}",
                    "efficiency": float(result["sparsity"]),
                    "approximation_gap": float(result["approximation_gap"]),
                }
            )
        for index, result in enumerate(sliding_variants):
            rows.append(
                {
                    "family": "sliding_window",
                    "variant": f"sliding_{index}",
                    "efficiency": float(result["cache_reduction"]),
                    "approximation_gap": float(result["approximation_gap"]),
                }
            )

        efficiency = np.array([row["efficiency"] for row in rows], dtype=float)
        gap = np.array([row["approximation_gap"] for row in rows], dtype=float)
        return {
            "variant_count": len(rows),
            "mean_efficiency": float(efficiency.mean()),
            "mean_approximation_gap": float(gap.mean()),
            "best_efficiency": float(efficiency.max()),
            "best_gap": float(gap.min()),
            "benchmark_rows": rows,
        }
