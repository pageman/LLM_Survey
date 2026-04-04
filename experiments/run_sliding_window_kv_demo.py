"""Run the KV-cache-aware sliding-window sparse attention demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import SlidingWindowKVDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = SlidingWindowKVDemo().evaluate()
    report = build_report(
        experiment_id="sliding_window_kv_demo",
        module="systems.sliding_window_kv_demo",
        metrics={
            "cache_reduction": result["cache_reduction"],
            "approximation_gap": result["approximation_gap"],
            "sliding_cache_tokens": result["sliding_cache_tokens"],
        },
        artifacts=result,
        notes=["Sliding-window sparse attention demo with KV-cache-aware token accounting."],
    )
    write_report(report, generated / "sliding_window_kv_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
