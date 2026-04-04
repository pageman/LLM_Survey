"""Run the sparse-attention masking demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, write_report
from src.modules.systems import SparseAttentionDemo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    generated = repo_root / "artifacts" / "generated"
    result = SparseAttentionDemo().evaluate()
    report = build_report(
        experiment_id="sparse_attention_demo",
        module="systems.sparse_attention_demo",
        metrics={
            "sparsity": result["sparsity"],
            "approximation_gap": result["approximation_gap"],
            "mask_density": result["mask_density"],
        },
        artifacts=result,
        notes=["Block-sparse local-plus-global masking demo for long-context intuition in pure NumPy."],
    )
    write_report(report, generated / "sparse_attention_demo.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
