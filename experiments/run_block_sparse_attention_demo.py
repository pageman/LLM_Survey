"""Run the block-sparse attention demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import BlockSparseAttentionDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = BlockSparseAttentionDemo().evaluate()
    report = build_report(
        experiment_id="block_sparse_attention_demo",
        module="systems.block_sparse_attention_demo",
        metrics={
            "mask_density": result["mask_density"],
            "approximation_gap": result["approximation_gap"],
        },
        artifacts=result,
        notes=["Block-sparse attention demo with coarse visible-block masking under a causal constraint."],
    )
    write_report(report, generated / "block_sparse_attention_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
