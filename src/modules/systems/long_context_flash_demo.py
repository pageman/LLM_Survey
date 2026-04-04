"""Long-context comparison between dense and tiled flash-style attention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import create_causal_mask, flash_attention_lite, scaled_dot_product_attention


@dataclass
class LongContextFlashDemo:
    seq_len: int = 48
    d_model: int = 16
    block_size: int = 8
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))
        mask = create_causal_mask(self.seq_len)

        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=mask)
        flash_out, diagnostics = flash_attention_lite(query, key, value, mask=mask, block_size=self.block_size)

        row_error = np.mean(np.abs(dense_out - flash_out), axis=1)
        dense_score_elements = self.seq_len * self.seq_len
        peak_tiled_score_elements = int(diagnostics["materialized_score_elements"])
        return {
            "seq_len": self.seq_len,
            "block_size": self.block_size,
            "mean_abs_error": float(row_error.mean()),
            "max_abs_error": float(np.max(np.abs(dense_out - flash_out))),
            "edge_error": float(np.mean(np.concatenate([row_error[:4], row_error[-4:]]))),
            "middle_error": float(np.mean(row_error[self.seq_len // 2 - 2 : self.seq_len // 2 + 2])),
            "dense_score_elements": dense_score_elements,
            "peak_tiled_score_elements": peak_tiled_score_elements,
            "dense_to_tiled_ratio": dense_score_elements / max(peak_tiled_score_elements, 1),
            "row_error_profile": row_error.round(8).tolist(),
        }
