"""Minimal smoke test for the installable package surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_survey


def main() -> None:
    tokenizer = llm_survey.ToyTokenizer.from_texts(["hello world", "educational llm survey"])
    tokens = tokenizer.encode("hello world")
    attention = llm_survey.MultiHeadAttention(d_model=8, num_heads=2, rng=np.random.default_rng(0))

    print("version", llm_survey.__version__)
    print("token_count", len(tokens))
    print("attention_heads", attention.num_heads)
    print("schema_version", llm_survey.SCHEMA_VERSION)


if __name__ == "__main__":
    main()
