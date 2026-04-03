"""Tiny causal language model built on the decoder-only transformer demo."""

from __future__ import annotations

import numpy as np

from src.core import ToyTokenizer, cross_entropy_from_probs, make_next_token_pairs, perplexity_from_losses
from src.modules.foundations import DecoderOnlyTransformerDemo


class CausalLanguageModel:
    """Educational next-token model using a decoder-only transformer forward path."""

    def __init__(
        self,
        tokenizer: ToyTokenizer,
        d_model: int = 16,
        num_heads: int = 2,
        d_ff: int = 32,
        num_layers: int = 1,
        max_seq_len: int = 64,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.model = DecoderOnlyTransformerDemo(
            vocab_size=len(tokenizer.vocab),
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            seed=seed,
        )

    def next_token_distribution(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            raise ValueError("input text must contain at least one token")
        return self.model.predict_next_distribution(token_ids)

    def score_text(self, text: str) -> dict[str, object]:
        token_ids = self.tokenizer.encode(text)
        inputs, targets = make_next_token_pairs(token_ids)
        losses = []

        for end_idx, target in enumerate(targets, start=1):
            prefix = inputs[:end_idx]
            probs = self.model.predict_next_distribution(prefix)
            losses.append(cross_entropy_from_probs(probs, target))

        return {
            "token_ids": token_ids,
            "losses": losses,
            "mean_loss": float(np.mean(losses)),
            "perplexity": perplexity_from_losses(losses),
        }

    def generate(self, prompt: str, max_new_tokens: int = 5) -> str:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [self.tokenizer.vocab[self.tokenizer.unk_token]]

        for _ in range(max_new_tokens):
            probs = self.model.predict_next_distribution(token_ids)
            next_token = int(np.argmax(probs))
            token_ids.append(next_token)

        return self.tokenizer.decode(token_ids)
