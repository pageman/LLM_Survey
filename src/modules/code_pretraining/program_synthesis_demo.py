"""Lite program-synthesis demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgramSynthesisDemo:
    def evaluate(self) -> dict[str, object]:
        spec_success = np.array([1, 1, 1, 1], dtype=float)
        program_success = np.array([1, 1, 0, 1], dtype=float)
        return {
            "spec_success": spec_success.astype(int).tolist(),
            "program_success": program_success.astype(int).tolist(),
            "exact_match": float((spec_success == program_success).mean()),
            "execution_success": float(program_success.mean()),
        }
