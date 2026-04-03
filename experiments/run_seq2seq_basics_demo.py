from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.foundations import Seq2SeqBasicsDemo


def main() -> None:
    output_dir = Path("artifacts/generated")
    result = Seq2SeqBasicsDemo().evaluate([1, 2, 3, 4])
    report = build_report("seq2seq_basics_demo", "foundations.seq2seq_basics", {"sequence_accuracy": result["sequence_accuracy"]}, result, ["Toy seq2seq reversal task."])
    write_report(report, output_dir / "seq2seq_basics_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
