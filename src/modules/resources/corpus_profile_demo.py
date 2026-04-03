"""Dedicated corpus-profile demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorpusProfileDemo:
    def evaluate(self) -> dict[str, object]:
        domains = {
            "web": 0.42,
            "books": 0.16,
            "code": 0.18,
            "science": 0.11,
            "dialogue": 0.08,
            "multilingual": 0.05,
        }
        distribution = np.array(list(domains.values()), dtype=float)
        entropy = float(-(distribution * np.log(distribution)).sum())
        return {
            "domain_entropy": entropy,
            "code_fraction": domains["code"],
            "multilingual_fraction": domains["multilingual"],
            "domains": domains,
        }
