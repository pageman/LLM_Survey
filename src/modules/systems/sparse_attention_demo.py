"""Block-sparse attention masking demo for long-context intuition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import scaled_dot_product_attention


@dataclass
class SparseAttentionDemo:
    seq_len: int = 16
    d_model: int = 8
    local_window: int = 2
    global_stride: int = 4
    seed: int = 0

    def _build_mask(self) -> np.ndarray:
        """Return a causal sparse mask where `0.0` means visible and non-zero means blocked."""
        rows = np.arange(self.seq_len)[:, None]
        cols = np.arange(self.seq_len)[None, :]
        local = np.abs(rows - cols) <= self.local_window
        global_token = (cols % self.global_stride) == 0
        causal_ok = cols <= rows
        visible = causal_ok & (local | global_token)
        return np.where(visible, 0.0, 1.0)

    def evaluate(self) -> dict[str, object]:
        """Return sparse-vs-dense comparison metrics under the shared mask convention."""
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))

        sparse_mask = self._build_mask()
        sparse_out, sparse_weights = scaled_dot_product_attention(query, key, value, mask=sparse_mask)
        dense_mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=float), k=1)
        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=dense_mask)

        density = float(np.mean(sparse_mask == 0.0))
        approximation_gap = float(np.mean(np.abs(dense_out - sparse_out)))
        return {
            "mask_density": density,
            "sparsity": 1.0 - density,
            "approximation_gap": approximation_gap,
            "local_window": self.local_window,
            "global_stride": self.global_stride,
            "mask_semantics": "0.0 visible, non-zero blocked",
            "attended_per_row": (sparse_mask == 0.0).sum(axis=1).astype(int).tolist(),
            "first_row_weights": sparse_weights[0].round(6).tolist(),
        }
