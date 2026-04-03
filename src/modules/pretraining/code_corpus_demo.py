"""Lite code-corpus composition demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CodeCorpusDemo:
    def evaluate(self) -> dict[str, object]:
        corpus_mix = np.array([0.34, 0.27, 0.18, 0.12, 0.09], dtype=float)
        syntax_signal = np.array([0.81, 0.78, 0.69, 0.73, 0.7], dtype=float)
        return {
            "corpus_mix": corpus_mix.tolist(),
            "syntax_signal": syntax_signal.tolist(),
            "code_coverage": float(corpus_mix.sum()),
            "syntax_density": float((corpus_mix * syntax_signal).sum()),
        }
