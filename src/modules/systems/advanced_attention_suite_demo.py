"""Richer advanced-core benchmark suite for normalization and tiled attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from src.core import (
    create_causal_mask,
    flash_attention_lite,
    layer_norm,
    online_softmax,
    scaled_dot_product_attention,
    stable_softmax,
    welford_layer_norm,
)


class AdvancedAttentionRow(TypedDict):
    seq_len: int
    block_size: int
    input_scale: float
    dense_flash_gap: float
    stable_online_gap: float
    dense_online_gap: float
    layer_norm_gap: float
    dense_to_tiled_ratio: float


@dataclass
class AdvancedAttentionSuiteDemo:
    d_model: int = 16
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        rows: list[AdvancedAttentionRow] = []
        for seq_len, block_size, scale in [(16, 4, 10.0), (32, 8, 100.0), (64, 8, 300.0), (96, 12, 500.0)]:
            query = rng.standard_normal((seq_len, self.d_model)) * scale
            key = rng.standard_normal((seq_len, self.d_model)) * scale
            value = rng.standard_normal((seq_len, self.d_model))
            mask = create_causal_mask(seq_len)

            dense_out, dense_weights = scaled_dot_product_attention(query, key, value, mask=mask)
            flash_out, diagnostics = flash_attention_lite(query, key, value, mask=mask, block_size=block_size)
            scores = np.einsum("qd,kd->qk", query, key, optimize=True) / np.sqrt(self.d_model)
            masked_scores = scores + (mask * -1e9)
            stable_weights = stable_softmax(masked_scores, axis=-1)
            online_weights = online_softmax(masked_scores, axis=-1)
            activations = rng.standard_normal((2, seq_len, self.d_model)) * scale + scale
            norm_reference = layer_norm(activations)
            norm_welford = welford_layer_norm(activations)

            rows.append(
                {
                    "seq_len": seq_len,
                    "block_size": block_size,
                    "input_scale": scale,
                    "dense_flash_gap": float(np.mean(np.abs(dense_out - flash_out))),
                    "stable_online_gap": float(np.max(np.abs(stable_weights - online_weights))),
                    "dense_online_gap": float(np.max(np.abs(dense_weights - online_weights))),
                    "layer_norm_gap": float(np.max(np.abs(norm_reference - norm_welford))),
                    "dense_to_tiled_ratio": float((seq_len * seq_len) / max(int(diagnostics["materialized_score_elements"]), 1)),
                }
            )

        return {
            "rows": rows,
            "max_dense_flash_gap": max(float(row["dense_flash_gap"]) for row in rows),
            "max_stable_online_gap": max(float(row["stable_online_gap"]) for row in rows),
            "max_layer_norm_gap": max(float(row["layer_norm_gap"]) for row in rows),
            "best_dense_to_tiled_ratio": max(float(row["dense_to_tiled_ratio"]) for row in rows),
            "validated_helpers": [
                "src/core/ops.py:online_softmax",
                "src/core/ops.py:stable_softmax",
                "src/core/ops.py:welford_layer_norm",
                "src/core/ops.py:layer_norm",
                "src/core/flash_attention_lite.py:flash_attention_lite",
            ],
        }
