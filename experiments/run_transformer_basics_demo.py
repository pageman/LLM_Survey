"""Local decoder-only transformer demo.

Usage:
    python3 experiments/run_transformer_basics_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.foundations import DecoderOnlyTransformerDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DecoderOnlyTransformerDemo(
        vocab_size=7,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=16,
        seed=0,
    )
    tokens = [0, 1, 2, 3, 4]
    logits, attention_maps = model.forward(tokens)
    probs = model.predict_next_distribution(tokens)

    result = build_report(
        experiment_id="transformer_basics_demo",
        module="foundations.transformer_basics",
        metrics={"logits_rows": int(logits.shape[0]), "logits_cols": int(logits.shape[1])},
        artifacts={
            "input_tokens": tokens,
            "next_token_distribution": probs.tolist(),
            "attention_shapes": [list(map(int, att.shape)) for att in attention_maps],
        },
        notes=["Decoder-only transformer forward-pass demo with causal masking."],
    )
    write_report(result, output_dir / "transformer_basics_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
