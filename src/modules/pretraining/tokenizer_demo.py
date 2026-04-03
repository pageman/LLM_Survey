"""Toy tokenizer comparison demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenizerDemo:
    def evaluate(self, text: str) -> dict[str, object]:
        word_tokens = text.split()
        char_tokens = list(text.replace(" ", ""))
        return {
            "text": text,
            "word_token_count": len(word_tokens),
            "char_token_count": len(char_tokens),
            "compression_ratio": len(word_tokens) / max(len(char_tokens), 1),
        }
