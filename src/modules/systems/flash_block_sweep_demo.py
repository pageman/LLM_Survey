"""Sweep flash-attention-lite block sizes against dense attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from src.core import create_causal_mask, flash_attention_lite, scaled_dot_product_attention


class BlockSweepRow(TypedDict):
    block_size: int
    mean_abs_error: float
    max_abs_error: float
    peak_materialized: int
    dense_to_tiled_ratio: float


@dataclass
class FlashBlockSweepDemo:
    seq_len: int = 32
    d_model: int = 16
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        query = rng.standard_normal((self.seq_len, self.d_model))
        key = rng.standard_normal((self.seq_len, self.d_model))
        value = rng.standard_normal((self.seq_len, self.d_model))
        mask = create_causal_mask(self.seq_len)
        dense_out, _ = scaled_dot_product_attention(query, key, value, mask=mask)
        dense_score_elements = self.seq_len * self.seq_len

        rows: list[BlockSweepRow] = []
        for block_size in [2, 4, 8, 16]:
            flash_out, diagnostics = flash_attention_lite(query, key, value, mask=mask, block_size=block_size)
            rows.append(
                {
                    "block_size": block_size,
                    "mean_abs_error": float(np.mean(np.abs(dense_out - flash_out))),
                    "max_abs_error": float(np.max(np.abs(dense_out - flash_out))),
                    "peak_materialized": int(diagnostics["materialized_score_elements"]),
                    "dense_to_tiled_ratio": dense_score_elements / max(int(diagnostics["materialized_score_elements"]), 1),
                }
            )
        best_ratio = max(row["dense_to_tiled_ratio"] for row in rows)
        best_error = min(row["mean_abs_error"] for row in rows)
        return {
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "best_dense_to_tiled_ratio": best_ratio,
            "best_mean_abs_error": best_error,
            "rows": rows,
        }
