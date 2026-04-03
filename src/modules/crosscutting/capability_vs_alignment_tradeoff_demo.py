"""Lite capability-versus-alignment tradeoff demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapabilityAlignmentTradeoffDemo:
    def evaluate(self) -> dict[str, object]:
        capability = np.array([0.63, 0.74, 0.82, 0.87], dtype=float)
        alignment = np.array([0.89, 0.84, 0.76, 0.69], dtype=float)
        return {
            "capability": capability.tolist(),
            "alignment": alignment.tolist(),
            "integration_score": float((capability * alignment).mean()),
            "tradeoff_correlation": float(np.corrcoef(capability, alignment)[0, 1]),
        }
