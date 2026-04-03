"""Local fine-tuning demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.adaptation import FineTuningExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    texts = [
        "biology cells contain dna information",
        "biology proteins fold into structures",
        "biology enzymes catalyze reactions",
    ]
    tokenizer = ToyTokenizer.from_texts(texts)
    experiment = FineTuningExperiment(tokenizer=tokenizer, hidden_size=12, learning_rate=0.05, seed=0)
    result = experiment.adapt(
        train_text="biology cells contain dna information",
        eval_text="biology cells contain dna information",
        steps=40,
    )
    report = build_report(
        experiment_id="finetuning_demo",
        module="adaptation.finetuning",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts={"train_loss_history": result["train_loss_history"]},
        notes=["Toy domain adaptation via direct fine-tuning on a tiny sequence corpus."],
    )
    write_report(report, output_dir / "finetuning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
