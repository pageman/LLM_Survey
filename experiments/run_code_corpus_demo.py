"""Local code-corpus pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import CodeCorpusDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = CodeCorpusDemo().evaluate()
    report = build_report(
        experiment_id="code_corpus_demo",
        module="pretraining.code_corpus_demo",
        metrics={"code_coverage": result["code_coverage"], "syntax_density": result["syntax_density"]},
        artifacts=result,
        notes=["Lite code-corpus demo over corpus composition and syntax signal."],
    )
    write_report(report, output_dir / "code_corpus_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
