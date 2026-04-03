"""Vanilla recurrent language model core.

Primary donor notebook:
- 02_char_rnn_karpathy.ipynb
"""

from __future__ import annotations

import numpy as np

from .attention import softmax


def one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros((size, 1), dtype=float)
    vector[index, 0] = 1.0
    return vector


class VanillaRNNLanguageModel:
    """Minimal character/token-level RNN language model with BPTT."""

    def __init__(self, vocab_size: int, hidden_size: int, rng: np.random.Generator | None = None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rng = rng or np.random.default_rng()

        scale = 0.01
        self.Wxh = self.rng.standard_normal((hidden_size, vocab_size)) * scale
        self.Whh = self.rng.standard_normal((hidden_size, hidden_size)) * scale
        self.Why = self.rng.standard_normal((vocab_size, hidden_size)) * scale
        self.bh = np.zeros((hidden_size, 1), dtype=float)
        self.by = np.zeros((vocab_size, 1), dtype=float)

    def initial_hidden(self) -> np.ndarray:
        return np.zeros((self.hidden_size, 1), dtype=float)

    def forward(
        self,
        inputs: list[int],
        hprev: np.ndarray | None = None,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev) if hprev is not None else self.initial_hidden()

        for t, token_idx in enumerate(inputs):
            xs[t] = one_hot(token_idx, self.vocab_size)
            hs[t] = np.tanh((self.Wxh @ xs[t]) + (self.Whh @ hs[t - 1]) + self.bh)
            ys[t] = (self.Why @ hs[t]) + self.by
            ps[t] = softmax(ys[t], axis=0)

        return xs, hs, ys, ps

    def loss(self, probabilities: dict[int, np.ndarray], targets: list[int]) -> float:
        total_loss = 0.0
        for t, target_idx in enumerate(targets):
            total_loss += -np.log(probabilities[t][target_idx, 0] + 1e-12)
        return float(total_loss)

    def backward(
        self,
        xs: dict[int, np.ndarray],
        hs: dict[int, np.ndarray],
        ps: dict[int, np.ndarray],
        targets: list[int],
        clip_value: float = 5.0,
    ) -> dict[str, np.ndarray]:
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0
            dWhy += dy @ hs[t].T
            dby += dy

            dh = (self.Why.T @ dy) + dhnext
            dhraw = (1.0 - hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t - 1].T
            dhnext = self.Whh.T @ dhraw

        grads = {
            "Wxh": dWxh,
            "Whh": dWhh,
            "Why": dWhy,
            "bh": dbh,
            "by": dby,
        }
        for grad in grads.values():
            np.clip(grad, -clip_value, clip_value, out=grad)
        return grads

    def sample(self, seed_ix: int, n: int, h: np.ndarray | None = None) -> list[int]:
        hidden = np.copy(h) if h is not None else self.initial_hidden()
        x = one_hot(seed_ix, self.vocab_size)
        indices = []

        for _ in range(n):
            hidden = np.tanh((self.Wxh @ x) + (self.Whh @ hidden) + self.bh)
            logits = (self.Why @ hidden) + self.by
            probs = softmax(logits, axis=0)
            ix = int(self.rng.choice(self.vocab_size, p=probs.ravel()))
            x = one_hot(ix, self.vocab_size)
            indices.append(ix)

        return indices
