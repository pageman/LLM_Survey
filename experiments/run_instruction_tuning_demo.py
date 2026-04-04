"""Local instruction-tuning demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.adaptation import InstructionTuningExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    train_pairs = [
        ("summarize article", "short summary"),
        ("translate hello", "hola"),
        ("classify sentiment positive", "positive"),
    ]
    eval_pair = ("translate hello", "hola")
    texts = []
    for instruction, response in train_pairs + [eval_pair]:
        texts.append(InstructionTuningExperiment.serialize_example(instruction, response))

    tokenizer = ToyTokenizer.from_texts(texts)
    experiment = InstructionTuningExperiment(tokenizer=tokenizer, hidden_size=16, learning_rate=0.05, seed=1)
    result = experiment.adapt(train_pairs=train_pairs, eval_pair=eval_pair, epochs=25)
    report = build_report(
        experiment_id="instruction_tuning_demo",
        module="adaptation.instruction_tuning",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts={
            "train_loss_history": result["train_loss_history"],
            "eval_example": list(eval_pair),
            "instruction_traces": result["instruction_traces"],
        },
        notes=["Instruction tuning with source-tagged instruction traces and before-after loss behavior."],
    )
    write_report(report, output_dir / "instruction_tuning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
