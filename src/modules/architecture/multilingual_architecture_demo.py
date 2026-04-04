"""Multilingual architecture demo with sharing and vocabulary-split structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultilingualArchitectureDemo:
    def evaluate(self) -> dict[str, object]:
        configurations = [
            {"configuration": "fully_shared", "language_pair": "en-es", "transfer": 0.79, "vocab_split": 0.10},
            {"configuration": "fully_shared", "language_pair": "en-de", "transfer": 0.74, "vocab_split": 0.18},
            {"configuration": "partially_shared", "language_pair": "en-tl", "transfer": 0.70, "vocab_split": 0.26},
            {"configuration": "language_specific_heads", "language_pair": "en-ar", "transfer": 0.67, "vocab_split": 0.31},
        ]
        language_transfer = np.array([item["transfer"] for item in configurations], dtype=float)
        return {
            "language_transfer": language_transfer.tolist(),
            "parameter_sharing_score": float(language_transfer.mean()),
            "transfer_score": float(language_transfer.min() / language_transfer.max()),
            "configurations": configurations,
        }
