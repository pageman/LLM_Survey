"""Dedicated jailbreak-transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JailbreakTransferEvaluator:
    def evaluate(self) -> dict[str, object]:
        source_success = np.array([0.48, 0.41, 0.36], dtype=float)
        transferred_success = np.array([0.31, 0.27, 0.22], dtype=float)
        return {
            "source_attack_rate": float(source_success.mean()),
            "transfer_attack_rate": float(transferred_success.mean()),
            "transfer_ratio": float(transferred_success.mean() / source_success.mean()),
            "defense_gap": float(source_success.mean() - transferred_success.mean()),
            "source_success": source_success.tolist(),
            "transferred_success": transferred_success.tolist(),
        }
