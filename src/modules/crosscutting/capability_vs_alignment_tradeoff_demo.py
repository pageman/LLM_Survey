"""Lite capability-versus-alignment tradeoff demo with frontier points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CapabilityAlignmentTradeoffDemo:
    def evaluate(self) -> dict[str, object]:
        settings = [
            {"setting": "base", "capability": 0.63, "alignment": 0.89},
            {"setting": "helpful_tuned", "capability": 0.74, "alignment": 0.84},
            {"setting": "capability_pushed", "capability": 0.82, "alignment": 0.76},
            {"setting": "aggressive_optimization", "capability": 0.87, "alignment": 0.69},
        ]
        capability = np.array([item["capability"] for item in settings], dtype=float)
        alignment = np.array([item["alignment"] for item in settings], dtype=float)
        frontiers = [
            {
                "setting": item["setting"],
                "capability": item["capability"],
                "alignment": item["alignment"],
                "frontier_score": float(item["capability"] * item["alignment"]),
            }
            for item in settings
        ]
        worst_alignment = min(settings, key=lambda item: item["alignment"])
        return {
            "capability": capability.tolist(),
            "alignment": alignment.tolist(),
            "integration_score": float((capability * alignment).mean()),
            "tradeoff_correlation": float(np.corrcoef(capability, alignment)[0, 1]),
            "frontiers": frontiers,
            "worst_alignment_setting": worst_alignment["setting"],
        }
