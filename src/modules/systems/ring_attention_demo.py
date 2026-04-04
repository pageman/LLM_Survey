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
        """Return a ring-style causal mask where local and previous shard tokens stay visible."""
        rows = np.arange(self.seq_len)[:, None]
        cols = np.arange(self.seq_len)[None, :]
        row_shard = rows // self.shard_size
        col_shard = cols // self.shard_size
        causal_ok = cols <= rows
        visible = causal_ok & ((col_shard == row_shard) | (col_shard == np.maximum(row_shard - 1, 0)))
        return np.where(visible, 0.0, 1.0)

    def evaluate(self) -> dict[str, object]:
        """Return ring-vs-dense comparison metrics under the shared mask convention."""
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
            "mask_semantics": "0.0 visible, non-zero blocked",
            "row_visibility": visible_per_row.tolist(),
            "first_row_weights": ring_weights[0].round(6).tolist(),
        }
