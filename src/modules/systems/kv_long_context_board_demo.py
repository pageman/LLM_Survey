"""KV-aware long-context board across efficient and sparse attention families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block_sparse_attention_demo import BlockSparseAttentionDemo
from .flash_attention_comparison_demo import FlashAttentionComparisonDemo
from .long_context_flash_demo import LongContextFlashDemo
from .ring_attention_demo import RingAttentionDemo
from .sliding_window_kv_demo import SlidingWindowKVDemo


@dataclass
class KVLongContextBoardDemo:
    def evaluate(self) -> dict[str, object]:
        flash_short = FlashAttentionComparisonDemo(seq_len=24, d_model=16, block_size=4, seed=1).evaluate()
        flash_long = LongContextFlashDemo(seq_len=64, d_model=16, block_size=8, seed=2).evaluate()
        sliding = SlidingWindowKVDemo(seq_len=48, d_model=8, window_size=8, sink_tokens=2, seed=3).evaluate()
        ring = RingAttentionDemo(seq_len=48, d_model=8, shard_size=8, seed=4).evaluate()
        block = BlockSparseAttentionDemo(seq_len=48, d_model=8, block_size=8, visible_blocks=2, seed=5).evaluate()

        rows = [
            {
                "family": "flash_short",
                "efficiency": float(1.0 - flash_short["memory_ratio_per_block"]),
                "approximation_gap": float(flash_short["mean_abs_error"]),
            },
            {
                "family": "flash_long",
                "efficiency": float(1.0 - (1.0 / max(flash_long["dense_to_tiled_ratio"], 1.0))),
                "approximation_gap": float(flash_long["mean_abs_error"]),
            },
            {
                "family": "sliding_window",
                "efficiency": float(sliding["cache_reduction"]),
                "approximation_gap": float(sliding["approximation_gap"]),
            },
            {
                "family": "ring",
                "efficiency": float(1.0 - ring["visibility_density"]),
                "approximation_gap": float(ring["approximation_gap"]),
            },
            {
                "family": "block_sparse",
                "efficiency": float(1.0 - block["mask_density"]),
                "approximation_gap": float(block["approximation_gap"]),
            },
        ]
        efficiency = np.array([row["efficiency"] for row in rows], dtype=float)
        gap = np.array([row["approximation_gap"] for row in rows], dtype=float)
        return {
            "rows": rows,
            "family_count": len(rows),
            "best_efficiency": float(efficiency.max()),
            "best_gap": float(gap.min()),
            "mean_efficiency": float(efficiency.mean()),
            "mean_gap": float(gap.mean()),
        }
