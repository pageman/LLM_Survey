"""Lite domain-coverage demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainCoverageDemo:
    def evaluate(self) -> dict[str, object]:
        domain_share = np.array([0.38, 0.24, 0.15, 0.1, 0.08, 0.05], dtype=float)
        tail_mass = float(domain_share[3:].sum())
        entropy = float(-(domain_share * np.log(domain_share)).sum())
        return {
            "domain_share": domain_share.tolist(),
            "tail_coverage": tail_mass,
            "domain_entropy": entropy,
        }
