"""Lite multilingual data-mixture demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultilingualDataDemo:
    def evaluate(self) -> dict[str, object]:
        token_share = np.array([0.50, 0.18, 0.12, 0.10, 0.06, 0.04], dtype=float)
        transfer = np.array([0.82, 0.74, 0.69, 0.63, 0.58, 0.54], dtype=float)
        return {
            "language_token_share": token_share.tolist(),
            "language_balance": float(token_share.min() / token_share.max()),
            "cross_lingual_transfer": float(transfer[1:].mean()),
            "transfer_scores": transfer.tolist(),
        }
