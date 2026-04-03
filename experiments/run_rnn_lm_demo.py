"""Local RNN LM demo.

Usage:
    python3 experiments/run_rnn_lm_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.foundations import RNNLanguageModel


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [0, 1, 2, 3, 0]
    targets = [1, 2, 3, 0, 1]

    model = RNNLanguageModel(vocab_size=4, hidden_size=8, learning_rate=0.05, seed=0)
    losses = []
    for _ in range(20):
        losses.append(model.train_step(inputs, targets))

    result = build_report(
        experiment_id="rnn_lm_demo",
        module="foundations.rnn_lm",
        metrics={"final_loss": losses[-1]},
        artifacts={"loss_history": losses, "sample": model.sample(seed_token=0, length=8)},
        notes=["Local smoke demo using the vanilla RNN core."],
    )
    write_report(result, output_dir / "rnn_lm_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
