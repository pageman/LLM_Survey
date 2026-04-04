"""Unified board across efficient and sparse attention families including quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block_sparse_attention_demo import BlockSparseAttentionDemo
from .flash_attention_comparison_demo import FlashAttentionComparisonDemo
from .kv_long_context_board_demo import KVLongContextBoardDemo
from .quantization_sim_demo import QuantizationSimDemo
from .ring_attention_demo import RingAttentionDemo
from .sliding_window_kv_demo import SlidingWindowKVDemo


@dataclass
class EfficientAttentionSuiteBoardDemo:
    def evaluate(self) -> dict[str, object]:
        flash = FlashAttentionComparisonDemo(seq_len=32, d_model=16, block_size=4, seed=1).evaluate()
        sliding = SlidingWindowKVDemo(seq_len=48, d_model=8, window_size=8, sink_tokens=2, seed=2).evaluate()
        ring = RingAttentionDemo(seq_len=48, d_model=8, shard_size=8, seed=3).evaluate()
        block = BlockSparseAttentionDemo(seq_len=48, d_model=8, block_size=8, visible_blocks=2, seed=4).evaluate()
        quant = QuantizationSimDemo(rows=16, cols=16, seed=5).evaluate()
        long_context = KVLongContextBoardDemo().evaluate()

        rows = [
            {"family": "flash", "efficiency": float(1.0 - flash["memory_ratio_per_block"]), "quality_gap": float(flash["mean_abs_error"])},
            {"family": "sliding_window", "efficiency": float(sliding["cache_reduction"]), "quality_gap": float(sliding["approximation_gap"])},
            {"family": "ring", "efficiency": float(1.0 - ring["visibility_density"]), "quality_gap": float(ring["approximation_gap"])},
            {"family": "block_sparse", "efficiency": float(1.0 - block["mask_density"]), "quality_gap": float(block["approximation_gap"])},
            {"family": "quantization", "efficiency": float(quant["int8_compression_ratio"] / 8.0), "quality_gap": float(quant["int8_mae"])},
        ]
        efficiencies = np.array([row["efficiency"] for row in rows], dtype=float)
        gaps = np.array([row["quality_gap"] for row in rows], dtype=float)
        return {
            "rows": rows,
            "family_count": len(rows),
            "best_efficiency": float(efficiencies.max()),
            "lowest_quality_gap": float(gaps.min()),
            "mean_efficiency": float(efficiencies.mean()),
            "mean_quality_gap": float(gaps.mean()),
            "long_context_reference": long_context,
        }
