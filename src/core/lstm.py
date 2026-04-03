"""LSTM primitives extracted and normalized from donor notebook code.

Primary donor notebook:
- 03_lstm_understanding.ipynb
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LSTMCell:
    """Single-step LSTM cell."""

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator | None = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rng = rng or np.random.default_rng()

        concat_size = input_size + hidden_size
        scale = 0.01
        self.Wf = self.rng.standard_normal((hidden_size, concat_size)) * scale
        self.bf = np.zeros((hidden_size, 1), dtype=float)
        self.Wi = self.rng.standard_normal((hidden_size, concat_size)) * scale
        self.bi = np.zeros((hidden_size, 1), dtype=float)
        self.Wc = self.rng.standard_normal((hidden_size, concat_size)) * scale
        self.bc = np.zeros((hidden_size, 1), dtype=float)
        self.Wo = self.rng.standard_normal((hidden_size, concat_size)) * scale
        self.bo = np.zeros((hidden_size, 1), dtype=float)

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        concat = np.vstack([x, h_prev])
        forget_gate = sigmoid((self.Wf @ concat) + self.bf)
        input_gate = sigmoid((self.Wi @ concat) + self.bi)
        candidate = np.tanh((self.Wc @ concat) + self.bc)
        c_next = (forget_gate * c_prev) + (input_gate * candidate)
        output_gate = sigmoid((self.Wo @ concat) + self.bo)
        h_next = output_gate * np.tanh(c_next)

        cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "concat": concat,
            "f": forget_gate,
            "i": input_gate,
            "c_tilde": candidate,
            "c_next": c_next,
            "o": output_gate,
            "h_next": h_next,
        }
        return h_next, c_next, cache


class LSTMSequenceModel:
    """Minimal sequence processor around `LSTMCell`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = rng or np.random.default_rng()
        self.cell = LSTMCell(input_size, hidden_size, rng=self.rng)
        self.Why = self.rng.standard_normal((output_size, hidden_size)) * 0.01
        self.by = np.zeros((output_size, 1), dtype=float)

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((self.hidden_size, 1), dtype=float),
            np.zeros((self.hidden_size, 1), dtype=float),
        )

    def forward(
        self,
        inputs: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], dict[str, list[np.ndarray]]]:
        h, c = self.initial_state()
        h_states = []
        c_states = []
        gate_values = {"f": [], "i": [], "o": []}

        for x in inputs:
            h, c, cache = self.cell.forward(x, h, c)
            h_states.append(h.copy())
            c_states.append(c.copy())
            gate_values["f"].append(cache["f"].copy())
            gate_values["i"].append(cache["i"].copy())
            gate_values["o"].append(cache["o"].copy())

        y = (self.Why @ h) + self.by
        return y, h_states, c_states, gate_values
