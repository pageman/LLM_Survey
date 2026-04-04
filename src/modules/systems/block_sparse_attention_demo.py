"""Block-sparse attention masking demo with coarse-grained block visibility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import scaled_dot_product_attention


@dataclass
class BlockSparseAttentionDemo:
    seq_len: int = 24
    d_model: int = 8
    block_size: int = 4
    visible_blocks: int = 2
    seed: int = 0

    def _mask(self) -> np.ndarray:
        """Return a block-sparse causal mask where `0.0` means visible."""
        rows = np.arange(self.seq_len)[:, None]
        cols = np.arange(self.seq_len)[None, :]
        row_blocks = rows // self.block_size
        col_blocks = cols // self.block_size
        causal_ok = cols <= rows
        within_visible_range = (col_blocks <= row_blocks) & (col_blocks >= (row_blocks - self.visible_blocks + 1))
        visible = causal_ok & within_visible_range
        return np.where(visible, 0.0, 1.0)

    def evaluate(self) -> dict[str, object]:
        """Return block-sparse-vs-dense comparison metrics under the shared mask convention."""
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))

        sparse_mask = self._mask()
        dense_mask = np.triu(np.ones((self.seq_len, self.seq_len), dtype=float), k=1)
        sparse_out, sparse_weights = scaled_dot_product_attention(query, key, value, mask=sparse_mask)
        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=dense_mask)

        density = float(np.mean(sparse_mask == 0.0))
        return {
            "block_size": self.block_size,
            "visible_blocks": self.visible_blocks,
            "mask_density": density,
            "approximation_gap": float(np.mean(np.abs(dense_out - sparse_out))),
            "mask_semantics": "0.0 visible, non-zero blocked",
            "visible_tokens_per_row": (sparse_mask == 0.0).sum(axis=1).astype(int).tolist(),
            "first_row_weights": sparse_weights[0].round(6).tolist(),
        }
