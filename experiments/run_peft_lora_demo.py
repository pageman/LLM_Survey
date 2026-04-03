"""Local PEFT/LoRA demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.adaptation import LoRALinearAdapterExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 4))
    target_W = np.array(
        [
            [1.5, -0.5, 0.2, 0.8],
            [-0.7, 1.2, 0.4, -0.3],
        ],
        dtype=float,
    )
    Y = X @ target_W.T

    experiment = LoRALinearAdapterExperiment(input_dim=4, output_dim=2, rank=1, learning_rate=0.2, seed=2)
    result = experiment.adapt(X, Y, steps=100)
    report = build_report(
        experiment_id="peft_lora_demo",
        module="adaptation.peft_lora",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
            "trainable_fraction": result["trainable_fraction"],
        },
        artifacts={"train_loss_history": result["train_loss_history"]},
        notes=["Toy low-rank adaptation with frozen base weights and trainable A/B factors."],
    )
    write_report(report, output_dir / "peft_lora_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
