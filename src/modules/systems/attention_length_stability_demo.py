"""Compare dense, tiled, and online-softmax attention across sequence lengths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from src.core import create_causal_mask, flash_attention_lite, online_softmax, scaled_dot_product_attention


class LengthRow(TypedDict):
    seq_len: int
    dense_flash_gap: float
    online_dense_weight_gap: float
    dense_peak_elements: int
    flash_peak_elements: int


@dataclass
class AttentionLengthStabilityDemo:
    d_model: int = 16
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        rows: list[LengthRow] = []
        for seq_len in [8, 16, 32, 64]:
            query = rng.standard_normal((seq_len, self.d_model))
            key = rng.standard_normal((seq_len, self.d_model))
            value = rng.standard_normal((seq_len, self.d_model))
            mask = create_causal_mask(seq_len)

            dense_out, dense_weights = scaled_dot_product_attention(query, key, value, mask=mask)
            flash_out, diagnostics = flash_attention_lite(query, key, value, mask=mask, block_size=max(4, seq_len // 8))

            raw_scores = np.einsum("qd,kd->qk", query, key, optimize=True) / np.sqrt(self.d_model)
            masked_scores = raw_scores + (mask * -1e9)
            online_weights = online_softmax(masked_scores, axis=-1)

            rows.append(
                {
                    "seq_len": seq_len,
                    "dense_flash_gap": float(np.mean(np.abs(dense_out - flash_out))),
                    "online_dense_weight_gap": float(np.max(np.abs(online_weights - dense_weights))),
                    "dense_peak_elements": int(seq_len * seq_len),
                    "flash_peak_elements": int(diagnostics["materialized_score_elements"]),
                }
            )

        return {
            "rows": rows,
            "max_dense_flash_gap": max(row["dense_flash_gap"] for row in rows),
            "max_online_dense_weight_gap": max(row["online_dense_weight_gap"] for row in rows),
            "best_memory_ratio": max(
                row["dense_peak_elements"] / max(row["flash_peak_elements"], 1)
                for row in rows
            ),
        }
