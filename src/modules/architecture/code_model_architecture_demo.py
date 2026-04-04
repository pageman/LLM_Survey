"""Code-model architecture demo with token, syntax, and identifier structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeModelArchitectureDemo:
    def evaluate(self) -> dict[str, object]:
        tasks = [
            {"task": "token_completion", "token_branch": 0.61, "ast_branch": 0.56, "identifier_branch": 0.58},
            {"task": "syntax_repair", "token_branch": 0.67, "ast_branch": 0.74, "identifier_branch": 0.69},
            {"task": "semantic_refactor", "token_branch": 0.71, "ast_branch": 0.79, "identifier_branch": 0.76},
        ]
        token_branch = np.array([item["token_branch"] for item in tasks], dtype=float)
        ast_branch = np.array([item["ast_branch"] for item in tasks], dtype=float)
        identifier_branch = np.array([item["identifier_branch"] for item in tasks], dtype=float)
        return {
            "token_branch": token_branch.tolist(),
            "ast_branch": ast_branch.tolist(),
            "syntax_bias_gain": float((ast_branch - token_branch).mean()),
            "code_structure_score": float(ast_branch.mean()),
            "identifier_bias_score": float(identifier_branch.mean()),
            "tasks": tasks,
        }
