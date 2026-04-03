"""Local causal LM demo.

Usage:
    python3 experiments/run_causal_lm_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.pretraining import CausalLanguageModel


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "large language models predict next tokens",
        "language models use causal masking",
        "transformers model token dependencies",
    ]
    tokenizer = ToyTokenizer.from_texts(texts)
    model = CausalLanguageModel(tokenizer=tokenizer, d_model=12, num_heads=2, d_ff=24, num_layers=1, seed=0)

    prompt = "large language models"
    score = model.score_text("large language models predict next tokens")
    generated = model.generate(prompt, max_new_tokens=3)

    result = build_report(
        experiment_id="causal_lm_demo",
        module="pretraining.causal_lm",
        metrics={"mean_loss": score["mean_loss"], "perplexity": score["perplexity"]},
        artifacts={"prompt": prompt, "generated": generated},
        notes=["Toy decoder-only next-token scoring and greedy generation demo."],
    )
    write_report(result, output_dir / "causal_lm_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
