"""Toy fine-tuning experiment built on the trainable RNN LM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core import ToyTokenizer, make_next_token_pairs
from src.modules.foundations import RNNLanguageModel


@dataclass
class FineTuningExperiment:
    tokenizer: ToyTokenizer
    hidden_size: int = 12
    learning_rate: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        self.model = RNNLanguageModel(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

    def loss_on_text(self, text: str) -> float:
        token_ids = self.tokenizer.encode(text)
        inputs, targets = make_next_token_pairs(token_ids)
        return self.model.evaluate_loss(inputs, targets)

    def adapt(self, train_text: str, eval_text: str, steps: int = 30) -> dict[str, object]:
        train_ids = self.tokenizer.encode(train_text)
        train_inputs, train_targets = make_next_token_pairs(train_ids)

        baseline_loss = self.loss_on_text(eval_text)
        losses = []
        best_loss = baseline_loss
        best_params = self.model.get_params()
        for _ in range(steps):
            losses.append(self.model.train_step(train_inputs, train_targets))
            eval_loss = self.loss_on_text(eval_text)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_params = self.model.get_params()

        self.model.set_params(best_params)
        adapted_loss = best_loss

        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "train_loss_history": losses,
        }
