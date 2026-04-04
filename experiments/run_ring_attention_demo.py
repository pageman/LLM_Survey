"""Run the ring-attention sparse masking demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import RingAttentionDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = RingAttentionDemo().evaluate()
    report = build_report(
        experiment_id="ring_attention_demo",
        module="systems.ring_attention_demo",
        metrics={
            "mean_visible_tokens": result["mean_visible_tokens"],
            "approximation_gap": result["approximation_gap"],
            "visibility_density": result["visibility_density"],
        },
        artifacts=result,
        notes=["Ring-attention style sparse masking demo over shard-local and previous-shard visibility."],
    )
    write_report(report, generated / "ring_attention_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
