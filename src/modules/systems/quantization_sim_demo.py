"""Pure-NumPy quantization simulation for educational inference demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizationSimDemo:
    rows: int = 12
    cols: int = 16
    seed: int = 0

    def evaluate(self) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        weights = rng.standard_normal((self.rows, self.cols))

        int8_scale = float(np.max(np.abs(weights)) / 127.0) if np.max(np.abs(weights)) > 0 else 1.0
        q_int8 = np.clip(np.round(weights / int8_scale), -127, 127).astype(np.int8)
        dequant_int8 = q_int8.astype(float) * int8_scale
        int8_mae = float(np.mean(np.abs(weights - dequant_int8)))

        fp8_scale = float(np.max(np.abs(weights)) / 15.0) if np.max(np.abs(weights)) > 0 else 1.0
        q_fp8 = np.clip(np.round(weights / fp8_scale), -15, 15).astype(np.int8)
        dequant_fp8 = q_fp8.astype(float) * fp8_scale
        fp8_mae = float(np.mean(np.abs(weights - dequant_fp8)))

        return {
            "int8_mae": int8_mae,
            "fp8_mae": fp8_mae,
            "int8_bits_per_weight": 8,
            "fp8_bits_per_weight": 8,
            "float_bits_per_weight": 64,
            "int8_compression_ratio": 64.0 / 8.0,
            "fp8_compression_ratio": 64.0 / 8.0,
            "sample_rows": {
                "original": weights[0, :6].round(4).tolist(),
                "int8": dequant_int8[0, :6].round(4).tolist(),
                "fp8": dequant_fp8[0, :6].round(4).tolist(),
            },
        }
