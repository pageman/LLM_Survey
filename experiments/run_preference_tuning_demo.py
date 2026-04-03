"""Local preference-tuning demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.adaptation import PreferenceTuningExperiment


def main() -> None:
    output_dir = Path("artifacts/generated")
    preferences = [
        ("answer safely", "provide a cautious response", "give dangerous instructions"),
        ("decline harmful request", "refuse and explain safety", "comply directly"),
        ("summarize neutrally", "give a balanced summary", "write a biased rant"),
    ]
    eval_preference = ("answer safely", "provide a cautious response", "give dangerous instructions")
    texts = []
    for prompt, chosen, rejected in preferences + [eval_preference]:
        texts.append(PreferenceTuningExperiment.serialize(prompt, chosen))
        texts.append(PreferenceTuningExperiment.serialize(prompt, rejected))

    tokenizer = ToyTokenizer.from_texts(texts)
    experiment = PreferenceTuningExperiment(tokenizer=tokenizer, hidden_size=16, learning_rate=0.05, seed=4)
    result = experiment.adapt(preferences=preferences, eval_preference=eval_preference, epochs=25)

    report = build_report(
        experiment_id="preference_tuning_demo",
        module="adaptation.preference_tuning",
        metrics={
            "baseline_loss": result["baseline_loss"],
            "adapted_loss": result["adapted_loss"],
            "gain": result["gain"],
        },
        artifacts={
            "baseline_margin": result["baseline_margin"],
            "adapted_margin": result["adapted_margin"],
            "train_loss_history": result["train_loss_history"],
        },
        notes=["Minimal pairwise preference tuning where chosen responses should outrank rejected ones."],
    )
    write_report(report, output_dir / "preference_tuning_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
