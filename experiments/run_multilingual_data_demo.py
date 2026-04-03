"""Local multilingual-data pretraining demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import MultilingualDataDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = MultilingualDataDemo().evaluate()
    report = build_report(
        experiment_id="multilingual_data_demo",
        module="pretraining.multilingual_data_demo",
        metrics={
            "language_balance": result["language_balance"],
            "cross_lingual_transfer": result["cross_lingual_transfer"],
        },
        artifacts=result,
        notes=["Lite multilingual data-mixture demo with token-share and transfer accounting."],
    )
    write_report(report, output_dir / "multilingual_data_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
