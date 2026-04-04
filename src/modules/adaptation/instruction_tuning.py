"""Toy instruction-tuning experiment using serialized prompt-response templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from src.core import ToyTokenizer, make_next_token_pairs
from src.modules.foundations import RNNLanguageModel

InstructionResponsePair: TypeAlias = tuple[str, str]


@dataclass
class InstructionTuningExperiment:
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
    def serialize_example(instruction: str, response: str) -> str:
        return f"instruction {instruction} response {response}"

    def loss_on_example(self, instruction: str, response: str) -> float:
        token_ids = self.tokenizer.encode(self.serialize_example(instruction, response))
        inputs, targets = make_next_token_pairs(token_ids)
        return self.model.evaluate_loss(inputs, targets)

    def adapt(
        self,
        train_pairs: list[InstructionResponsePair],
        eval_pair: InstructionResponsePair,
        epochs: int = 20,
    ) -> dict[str, object]:
        if not train_pairs:
            raise ValueError("train_pairs must not be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        baseline_loss = self.loss_on_example(*eval_pair)
        history = []
        instruction_traces = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for instruction, response in train_pairs:
                loss_before = self.loss_on_example(instruction, response)
                token_ids = self.tokenizer.encode(self.serialize_example(instruction, response))
                inputs, targets = make_next_token_pairs(token_ids)
                epoch_loss += self.model.train_step(inputs, targets)
                loss_after = self.loss_on_example(instruction, response)
                instruction_traces.append(
                    {
                        "instruction": instruction,
                        "response": response,
                        "loss_before": loss_before,
                        "loss_after": loss_after,
                        "instruction_source": "toy_supervision_set",
                        "epoch": epoch,
                    }
                )
            history.append(epoch_loss / max(len(train_pairs), 1))

        adapted_loss = self.loss_on_example(*eval_pair)
        return {
            "baseline_loss": baseline_loss,
            "adapted_loss": adapted_loss,
            "gain": baseline_loss - adapted_loss,
            "train_loss_history": history,
            "instruction_traces": instruction_traces[-len(train_pairs) :],
        }
