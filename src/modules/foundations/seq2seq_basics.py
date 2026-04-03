"""Toy seq2seq basics module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Seq2SeqBasicsDemo:
    def evaluate(self, source_tokens: list[int], reverse_output: bool = True) -> dict[str, object]:
        target = list(reversed(source_tokens)) if reverse_output else source_tokens[:]
        return {
            "source_tokens": source_tokens,
            "decoded_tokens": target,
            "sequence_accuracy": 1.0 if target == list(reversed(source_tokens)) else 0.0,
        }
