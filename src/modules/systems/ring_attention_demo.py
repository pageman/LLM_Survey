"""Ring-attention style local-ring masking demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import scaled_dot_product_attention


@dataclass
class RingAttentionDemo:
    seq_len: int = 24
    d_model: int = 8
    shard_size: int = 6
    seed: int = 0

    def _ring_mask(self) -> np.ndarray:
        mask = np.ones((self.seq_len, self.seq_len), dtype=float)
        for row in range(self.seq_len):
            shard = row // self.shard_size
            left = shard * self.shard_size
            right = min(self.seq_len, left + self.shard_size)
            prev_left = max(0, left - self.shard_size)
            visible = list(range(prev_left, right))
            visible = [col for col in visible if col <= row]
            mask[row, visible] = 0.0
        return mask

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))

        ring_mask = self._ring_mask()
        dense_mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=float), k=1)
        ring_out, ring_weights = scaled_dot_product_attention(query, key, value, mask=ring_mask)
        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=dense_mask)

        visible_per_row = (ring_mask == 0.0).sum(axis=1).astype(int)
        return {
            "shard_size": self.shard_size,
            "mean_visible_tokens": float(visible_per_row.mean()),
            "approximation_gap": float(np.mean(np.abs(dense_out - ring_out))),
            "visibility_density": float(np.mean(ring_mask == 0.0)),
            "row_visibility": visible_per_row.tolist(),
            "first_row_weights": ring_weights[0].round(6).tolist(),
        }
