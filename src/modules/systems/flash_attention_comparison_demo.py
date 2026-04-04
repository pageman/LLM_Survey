"""Compare tiled flash-attention-lite against full attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import create_causal_mask, flash_attention_lite, scaled_dot_product_attention


@dataclass
class FlashAttentionComparisonDemo:
    seq_len: int = 12
    d_model: int = 16
    block_size: int = 4
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))
        mask = create_causal_mask(self.seq_len)

        full_out, _ = scaled_dot_product_attention(query, key, value, mask=mask)
        flash_out, diagnostics = flash_attention_lite(
            query,
            key,
            value,
            mask=mask,
            block_size=self.block_size,
        )
        max_abs_error = float(np.max(np.abs(full_out - flash_out)))
        mean_abs_error = float(np.mean(np.abs(full_out - flash_out)))
        dense_elements = self.seq_len * self.seq_len

        return {
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "block_size": self.block_size,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "full_score_elements": dense_elements,
            "tiled_score_elements_per_block": diagnostics["materialized_score_elements"],
            "num_blocks": diagnostics["num_blocks"],
            "memory_ratio_per_block": diagnostics["materialized_score_elements"] / max(dense_elements, 1),
            "diagnostics": diagnostics,
        }
