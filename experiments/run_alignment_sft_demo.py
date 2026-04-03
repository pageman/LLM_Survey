"""Local alignment SFT demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.adaptation import AlignmentSFTExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    demonstrations = [
        ("answer safely", "provide a cautious response"),
        ("decline harmful request", "refuse and explain safety"),
        ("summarize neutrally", "give a balanced summary"),
    ]
    eval_pair = ("answer safely", "provide a cautious response")
    texts = [AlignmentSFTExperiment.serialize(prompt, chosen) for prompt, chosen in demonstrations + [eval_pair]]

    tokenizer = ToyTokenizer.from_texts(texts)
    experiment = AlignmentSFTExperiment(tokenizer=tokenizer, hidden_size=16, learning_rate=0.05, seed=3)
    result = experiment.adapt(demonstrations=demonstrations, eval_pair=eval_pair, epochs=25)

    report = build_report(
        experiment_id="alignment_sft_demo",
        module="adaptation.alignment_sft",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts={"train_loss_history": result["train_loss_history"], "eval_pair": list(eval_pair)},
        notes=["Minimal supervised alignment fine-tuning on preferred demonstration responses."],
    )
    write_report(report, output_dir / "alignment_sft_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
