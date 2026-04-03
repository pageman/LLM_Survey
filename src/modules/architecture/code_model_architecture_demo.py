"""Lite code-model architecture demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeModelArchitectureDemo:
    def evaluate(self) -> dict[str, object]:
        token_branch = np.array([0.61, 0.67, 0.71], dtype=float)
        ast_branch = np.array([0.56, 0.74, 0.79], dtype=float)
        return {
            "token_branch": token_branch.tolist(),
            "ast_branch": ast_branch.tolist(),
            "syntax_bias_gain": float((ast_branch - token_branch).mean()),
            "code_structure_score": float(ast_branch.mean()),
        }
