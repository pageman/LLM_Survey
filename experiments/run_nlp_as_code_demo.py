"""Local NLP-as-code code-pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.code_pretraining import NLPAsCodeDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = NLPAsCodeDemo().evaluate()
    report = build_report(
        experiment_id="nlp_as_code_demo",
        module="code_pretraining.nlp_as_code_demo",
        metrics={"compression_ratio": result["compression_ratio"], "structuring_gain": result["structuring_gain"]},
        artifacts=result,
        notes=["Lite NLP-as-code demo over compression and structuring effects."],
    )
    write_report(report, output_dir / "nlp_as_code_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
