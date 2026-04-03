from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.applications import ScientificAssistantDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = ScientificAssistantDemo().evaluate()
    report = build_report("scientific_assistant_demo", "applications.scientific_assistant_demo", {"hypothesis_quality": result["hypothesis_quality"]}, result, ["Toy scientific-assistant application demo."])
    write_report(report, output_dir / "scientific_assistant_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
