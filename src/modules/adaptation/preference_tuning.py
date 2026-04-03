"""Minimal preference-tuning experiment with pairwise chosen/rejected signals."""

from __future__ import annotations

from dataclasses import dataclass

from src.core import ToyTokenizer, make_next_token_pairs
from src.modules.foundations import RNNLanguageModel


@dataclass
class PreferenceTuningExperiment:
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
    def serialize(prompt: str, response: str) -> str:
        return f"prompt {prompt} response {response}"

    def loss_on_response(self, prompt: str, response: str) -> float:
        token_ids = self.tokenizer.encode(self.serialize(prompt, response))
        inputs, targets = make_next_token_pairs(token_ids)
        return self.model.evaluate_loss(inputs, targets)

    def preference_margin(self, prompt: str, chosen: str, rejected: str) -> float:
        chosen_loss = self.loss_on_response(prompt, chosen)
        rejected_loss = self.loss_on_response(prompt, rejected)
        return rejected_loss - chosen_loss

    def adapt(
        self,
        preferences: list[tuple[str, str, str]],
        eval_preference: tuple[str, str, str],
        epochs: int = 20,
    ) -> dict[str, object]:
        baseline_margin = self.preference_margin(*eval_preference)
        history = []

        for _ in range(epochs):
            epoch_signal = 0.0
            for prompt, chosen, rejected in preferences:
                chosen_ids = self.tokenizer.encode(self.serialize(prompt, chosen))
                chosen_inputs, chosen_targets = make_next_token_pairs(chosen_ids)
                epoch_signal += self.model.train_step(chosen_inputs, chosen_targets)
                # Penalize rejected completions indirectly by not training them and
                # tracking the margin on the chosen trajectory only.
            history.append(epoch_signal / max(len(preferences), 1))

        adapted_margin = self.preference_margin(*eval_preference)
        # Convert margin improvement into comparable loss-like metrics.
        baseline_loss = -baseline_margin
        adapted_loss = -adapted_margin
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "baseline_margin": baseline_margin,
            "adapted_margin": adapted_margin,
            "train_loss_history": history,
        }
