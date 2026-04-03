from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.pretraining import PrefixDecoderDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = PrefixDecoderDemo().evaluate("prefix prompt", "generated continuation")
    report = build_report("prefix_decoder_demo", "pretraining.prefix_decoder_demo", {"prefix_length": result["prefix_length"]}, result, ["Toy prefix-decoder serialization demo."])
    write_report(report, output_dir / "prefix_decoder_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
