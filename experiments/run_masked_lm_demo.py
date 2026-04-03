from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ToyTokenizer, build_report, write_report
from src.modules.pretraining import MaskedLMDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    corpus = [
        "transformers use masked language modeling",
        "bert uses masked language modeling",
        "modern transformers use self attention",
    ]
    tokenizer = ToyTokenizer.from_texts(corpus)
    result = MaskedLMDemo(tokenizer, corpus_texts=corpus).evaluate("transformers use masked language modeling")
    report = build_report("masked_lm_demo", "pretraining.masked_lm_demo", {"reconstruction_match": float(result["reconstruction_match"])}, result, ["Toy masked LM reconstruction demo."])
    write_report(report, output_dir / "masked_lm_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
