"""Lite program-aided reasoning demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgramAidedReasoningDemo:
    def evaluate(self) -> dict[str, object]:
        direct = np.array([0.41, 0.44, 0.39], dtype=float)
        executed = np.array([0.82, 0.86, 0.79], dtype=float)
        return {
            "direct": direct.tolist(),
            "program_aided": executed.tolist(),
            "execution_gain": float((executed - direct).mean()),
            "program_success": float(executed.mean()),
        }
