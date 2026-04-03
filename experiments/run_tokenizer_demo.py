from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import TokenizerDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = TokenizerDemo().evaluate("large language models")
    report = build_report("tokenizer_demo", "pretraining.tokenizer_demo", {"compression_ratio": result["compression_ratio"]}, result, ["Toy tokenizer granularity comparison."])
    write_report(report, output_dir / "tokenizer_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
