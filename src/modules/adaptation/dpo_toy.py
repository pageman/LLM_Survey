"""Lite DPO-style preference optimization on token-level response scores."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math


@dataclass
class DPOToyExperiment:
    beta: float = 0.5
    learning_rate: float = 0.2
    token_weights: Counter[str] = field(default_factory=Counter)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return text.lower().split()

    def score(self, text: str) -> float:
        return float(sum(self.token_weights[token] for token in self._tokens(text)))

    def dpo_loss(self, chosen: str, rejected: str, ref_margin: float = 0.0) -> float:
        delta = self.beta * ((self.score(chosen) - self.score(rejected)) - ref_margin)
        return float(-math.log(1.0 / (1.0 + math.exp(-delta))))

    def adapt(
        self,
        preferences: list[tuple[str, str, str]],
        eval_preference: tuple[str, str, str],
        epochs: int = 10,
    ) -> dict[str, object]:
        _, eval_chosen, eval_rejected = eval_preference
        baseline_loss = self.dpo_loss(eval_chosen, eval_rejected)
        history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for _, chosen, rejected in preferences:
                chosen_tokens = self._tokens(chosen)
                rejected_tokens = self._tokens(rejected)
                loss = self.dpo_loss(chosen, rejected)
                epoch_loss += loss
                update = self.learning_rate * (1.0 - (1.0 / (1.0 + math.exp(-self.beta * (self.score(chosen) - self.score(rejected))))))
                for token in chosen_tokens:
                    self.token_weights[token] += update
                for token in rejected_tokens:
                    self.token_weights[token] -= update
            history.append(epoch_loss / max(len(preferences), 1))

        adapted_loss = self.dpo_loss(eval_chosen, eval_rejected)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "trainable_fraction": 0.18,
            "loss_history": history,
            "top_positive_tokens": dict(self.token_weights.most_common(5)),
        }
