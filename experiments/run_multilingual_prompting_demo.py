"""Local multilingual prompting demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.multilingual.prompting_demo import MultilingualPromptingDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MultilingualPromptingDemo().evaluate()
    report = build_report(
        experiment_id="multilingual_prompting_demo",
        module="multilingual.prompting_demo",
        metrics={"native_prompt_score": result["native_prompt_score"], "translation_gap": result["translation_gap"]},
        artifacts=result,
        notes=["Lite multilingual prompting demo over native versus translated prompts."],
    )
    write_report(report, output_dir / "multilingual_prompting_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
