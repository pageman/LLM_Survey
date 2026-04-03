"""Dedicated library-stack matrix demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LibraryStackMatrix:
    def evaluate(self) -> dict[str, object]:
        matrix = [
            {"library": "NumPy", "core_math": 1, "autograd": 0, "serving": 0, "tokenization": 0},
            {"library": "PyTorch", "core_math": 1, "autograd": 1, "serving": 1, "tokenization": 0},
            {"library": "Transformers", "core_math": 0, "autograd": 0, "serving": 1, "tokenization": 1},
            {"library": "SentencePiece", "core_math": 0, "autograd": 0, "serving": 0, "tokenization": 1},
        ]
        coverage = sum(sum(int(value) for key, value in row.items() if key != "library") for row in matrix)
        return {
            "num_libraries": len(matrix),
            "capability_coverage": coverage,
            "matrix": matrix,
        }
