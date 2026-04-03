"""Small but more faithful masked language modeling demo.

The original placeholder simply returned the masked token. This version
builds a tiny context-count model from a toy corpus and predicts the masked
token from left/right context, which is much closer to the actual MLM idea:
recover a missing token from bidirectional context.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.core import ToyTokenizer


@dataclass
class MaskedLMDemo:
    tokenizer: ToyTokenizer
    corpus_texts: list[str] | None = None

    def __post_init__(self) -> None:
        corpus = self.corpus_texts or []
        self.context_counts: dict[tuple[str | None, str | None], Counter[str]] = defaultdict(Counter)
        self.unigram_counts: Counter[str] = Counter()

        for text in corpus:
            tokens = text.lower().split()
            for idx, token in enumerate(tokens):
                left = tokens[idx - 1] if idx > 0 else None
                right = tokens[idx + 1] if idx + 1 < len(tokens) else None
                self.context_counts[(left, right)][token] += 1
                self.unigram_counts[token] += 1

    def _predict_from_context(self, left: str | None, right: str | None) -> tuple[str, dict[str, int]]:
        candidates = self.context_counts.get((left, right))
        if not candidates:
            if left is not None:
                for (cand_left, _), counts in self.context_counts.items():
                    if cand_left == left:
                        candidates = candidates or Counter()
                        candidates.update(counts)
            if right is not None:
                for (_, cand_right), counts in self.context_counts.items():
                    if cand_right == right:
                        candidates = candidates or Counter()
                        candidates.update(counts)
        if not candidates:
            candidates = self.unigram_counts
        if not candidates:
            candidates = Counter({self.tokenizer.unk_token: 1})

        predicted_token, _ = candidates.most_common(1)[0]
        return predicted_token, dict(candidates.most_common(5))

    def evaluate(self, text: str, mask_token: str = "<unk>") -> dict[str, object]:
        tokens = text.lower().split()
        mask_index = min(1, len(tokens) - 1)
        original = tokens[mask_index]
        left = tokens[mask_index - 1] if mask_index > 0 else None
        right = tokens[mask_index + 1] if mask_index + 1 < len(tokens) else None
        tokens[mask_index] = mask_token
        reconstructed, top_candidates = self._predict_from_context(left, right)
        return {
            "masked_text": " ".join(tokens),
            "predicted_token": reconstructed,
            "reconstruction_match": reconstructed == original,
            "mask_index": mask_index,
            "left_context": left,
            "right_context": right,
            "top_candidates": top_candidates,
        }
