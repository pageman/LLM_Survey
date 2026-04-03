"""Shared data helpers for local toy corpora and token workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class ToyTokenizer:
    """Whitespace tokenizer with explicit vocab maps."""

    vocab: dict[str, int]
    inv_vocab: dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        extra_tokens: list[str] | None = None,
    ) -> "ToyTokenizer":
        extra_tokens = extra_tokens or []
        tokens = []
        for token in extra_tokens:
            if token not in tokens:
                tokens.append(token)

        for text in texts:
            for token in text.lower().split():
                if token not in tokens:
                    tokens.append(token)

        if "<unk>" not in tokens:
            tokens.insert(0, "<unk>")

        vocab = {token: idx for idx, token in enumerate(tokens)}
        inv_vocab = {idx: token for token, idx in vocab.items()}
        return cls(vocab=vocab, inv_vocab=inv_vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in text.lower().split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.inv_vocab.get(idx, self.unk_token) for idx in token_ids)


def make_next_token_pairs(token_ids: list[int]) -> tuple[list[int], list[int]]:
    if len(token_ids) < 2:
        raise ValueError("need at least two tokens to form next-token pairs")
    return token_ids[:-1], token_ids[1:]


def one_hot_sequence(token_ids: list[int], vocab_size: int) -> list[np.ndarray]:
    vectors = []
    for token_id in token_ids:
        vector = np.zeros((vocab_size, 1), dtype=float)
        vector[token_id, 0] = 1.0
        vectors.append(vector)
    return vectors


def hashed_bow_embedding(text: str, embedding_dim: int) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding."""
    vector = np.zeros((embedding_dim,), dtype=float)
    counts = Counter(text.lower().split())
    for token, count in counts.items():
        slot = hash(token) % embedding_dim
        sign = 1.0 if (hash((token, "sign")) % 2 == 0) else -1.0
        vector[slot] += sign * count

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector
