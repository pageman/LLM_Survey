"""Toy code generation application demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodeGenerationDemo:
    def evaluate(self) -> dict[str, object]:
        generated_code = "def add(a, b):\n    return a + b\n"
        pass_rate = 1.0
        return {
            "generated_code": generated_code,
            "pass_rate": pass_rate,
        }
