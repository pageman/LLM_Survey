"""Local LSTM LM demo.

Usage:
    python3 experiments/run_lstm_lm_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.foundations import LSTMLanguageModel


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LSTMLanguageModel(vocab_size=5, hidden_size=8, seed=0)
    tokens = [0, 1, 2, 3]
    probs = model.predict_next_distribution(tokens)
    next_token = model.sample_next(tokens)

    result = build_report(
        experiment_id="lstm_lm_demo",
        module="foundations.lstm_lm",
        metrics={"sampled_next_token": next_token},
        artifacts={"input_tokens": tokens, "next_token_distribution": probs.tolist()},
        notes=["Local forward-pass demo for the LSTM language model facade."],
    )
    write_report(result, output_dir / "lstm_lm_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
