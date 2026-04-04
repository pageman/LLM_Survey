"""Unified board across efficient and sparse attention families including quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from .advanced_attention_suite_demo import AdvancedAttentionSuiteDemo
from .block_sparse_attention_demo import BlockSparseAttentionDemo
from .flash_attention_comparison_demo import FlashAttentionComparisonDemo
from .kv_long_context_board_demo import KVLongContextBoardDemo
from .quantization_sim_demo import QuantizationSimDemo
from .ring_attention_demo import RingAttentionDemo
from .sliding_window_kv_demo import SlidingWindowKVDemo


class EfficientAttentionRow(TypedDict):
    family: str
    efficiency: float
    quality_gap: float
    evidence: str


@dataclass
class EfficientAttentionSuiteBoardDemo:
    def evaluate(self) -> dict[str, object]:
        flash = FlashAttentionComparisonDemo(seq_len=32, d_model=16, block_size=4, seed=1).evaluate()
        sliding = SlidingWindowKVDemo(seq_len=48, d_model=8, window_size=8, sink_tokens=2, seed=2).evaluate()
        ring = RingAttentionDemo(seq_len=48, d_model=8, shard_size=8, seed=3).evaluate()
        block = BlockSparseAttentionDemo(seq_len=48, d_model=8, block_size=8, visible_blocks=2, seed=4).evaluate()
        quant = QuantizationSimDemo(rows=16, cols=16, seed=5).evaluate()
        long_context = KVLongContextBoardDemo().evaluate()
        advanced_suite = AdvancedAttentionSuiteDemo(seed=6).evaluate()

        rows: list[EfficientAttentionRow] = [
            {
                "family": "flash",
                "efficiency": float(1.0 - flash["memory_ratio_per_block"]),
                "quality_gap": float(flash["mean_abs_error"]),
                "evidence": "flash_attention_lite peak memory ratio",
            },
            {
                "family": "sliding_window",
                "efficiency": float(sliding["cache_reduction"]),
                "quality_gap": float(sliding["approximation_gap"]),
                "evidence": "kv-aware cache reduction",
            },
            {
                "family": "ring",
                "efficiency": float(1.0 - ring["visibility_density"]),
                "quality_gap": float(ring["approximation_gap"]),
                "evidence": "ring visibility density",
            },
            {
                "family": "block_sparse",
                "efficiency": float(1.0 - block["mask_density"]),
                "quality_gap": float(block["approximation_gap"]),
                "evidence": "block visibility density",
            },
            {
                "family": "quantization",
                "efficiency": float(quant["int8_compression_ratio"] / 8.0),
                "quality_gap": float(quant["int8_mae"]),
                "evidence": "int8 compression ratio vs reconstruction error",
            },
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
            "canonical_summary_surface": True,
            "shared_metric_system": ["efficiency", "quality_gap"],
            "long_context_reference": long_context,
            "advanced_suite_reference": advanced_suite,
        }
