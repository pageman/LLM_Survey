"""Sliding-window sparse attention with KV-cache-aware accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import scaled_dot_product_attention


@dataclass
class SlidingWindowKVDemo:
    seq_len: int = 32
    d_model: int = 8
    window_size: int = 6
    sink_tokens: int = 2
    seed: int = 0

    def _mask(self) -> np.ndarray:
        mask = np.ones((self.seq_len, self.seq_len), dtype=float)
        for row in range(self.seq_len):
            left = max(0, row - self.window_size + 1)
            for col in range(left, row + 1):
                mask[row, col] = 0.0
            mask[row, : min(self.sink_tokens, row + 1)] = 0.0
        return mask

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))

        dense_mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=float), k=1)
        sliding_mask = self._mask()
        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=dense_mask)
        sliding_out, _ = scaled_dot_product_attention(query, key, value, mask=sliding_mask)

        dense_cache_tokens = self.seq_len * (self.seq_len + 1) // 2
        sliding_cache_tokens = int(np.sum(sliding_mask == 0.0))
        return {
            "window_size": self.window_size,
            "sink_tokens": self.sink_tokens,
            "dense_cache_tokens": dense_cache_tokens,
            "sliding_cache_tokens": sliding_cache_tokens,
            "cache_reduction": 1.0 - (sliding_cache_tokens / max(dense_cache_tokens, 1)),
            "approximation_gap": float(np.mean(np.abs(dense_out - sliding_out))),
            "row_token_counts": (sliding_mask == 0.0).sum(axis=1).astype(int).tolist(),
        }
