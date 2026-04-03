"""Minimal supervised alignment / SFT experiment."""

from __future__ import annotations

from dataclasses import dataclass

from src.core import ToyTokenizer, make_next_token_pairs
from src.modules.foundations import RNNLanguageModel


@dataclass
class AlignmentSFTExperiment:
    tokenizer: ToyTokenizer
    hidden_size: int = 16
    learning_rate: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        self.model = RNNLanguageModel(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

    @staticmethod
    def serialize(prompt: str, chosen_response: str) -> str:
        return f"prompt {prompt} aligned_response {chosen_response}"

    def loss_on_pair(self, prompt: str, chosen_response: str) -> float:
        token_ids = self.tokenizer.encode(self.serialize(prompt, chosen_response))
        inputs, targets = make_next_token_pairs(token_ids)
        return self.model.evaluate_loss(inputs, targets)

    def adapt(
        self,
        demonstrations: list[tuple[str, str]],
        eval_pair: tuple[str, str],
        epochs: int = 20,
    ) -> dict[str, object]:
        baseline_loss = self.loss_on_pair(*eval_pair)
        history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for prompt, chosen in demonstrations:
                token_ids = self.tokenizer.encode(self.serialize(prompt, chosen))
                inputs, targets = make_next_token_pairs(token_ids)
                epoch_loss += self.model.train_step(inputs, targets)
            history.append(epoch_loss / max(len(demonstrations), 1))

        adapted_loss = self.loss_on_pair(*eval_pair)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "train_loss_history": history,
        }
