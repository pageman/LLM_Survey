"""Local retrieval demo.

Usage:
    python3 experiments/run_retrieval_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import build_report, compute_retrieval_metrics, write_report
from src.modules.utilization import DenseRetriever, HybridRetriever, SimpleBM25Retriever


def main() -> None:
    output_dir = Path("artifacts/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        "paris is the capital of france",
        "tokyo is the capital of japan",
        "manila is the capital of the philippines",
        "berlin is the capital of germany",
    ]
    queries = [
        ("capital of france", 0),
        ("capital of japan", 1),
        ("capital of philippines", 2),
    ]

    dense = DenseRetriever(embedding_dim=64)
    bm25 = SimpleBM25Retriever(documents)
    hybrid = HybridRetriever(documents, embedding_dim=64, alpha=0.5)

    dense_predictions = []
    bm25_predictions = []
    hybrid_predictions = []
    correct = []
    for query, answer_idx in queries:
        dense_predictions.append(dense.retrieve(query, documents, k=3)[0].tolist())
        bm25_predictions.append(bm25.retrieve(query, k=3)[0].tolist())
        hybrid_predictions.append(hybrid.retrieve(query, k=3)[0].tolist())
        correct.append(answer_idx)

    dense_metrics = compute_retrieval_metrics(dense_predictions, correct, k_values=[1, 3])
    bm25_metrics = compute_retrieval_metrics(bm25_predictions, correct, k_values=[1, 3])
    hybrid_metrics = compute_retrieval_metrics(hybrid_predictions, correct, k_values=[1, 3])

    result = build_report(
        experiment_id="retrieval_demo",
        module="utilization.retrieval",
        metrics={
            "dense_mrr": dense_metrics[1],
            "bm25_mrr": bm25_metrics[1],
            "hybrid_mrr": hybrid_metrics[1],
        },
        artifacts={
            "dense_predictions": dense_predictions,
            "bm25_predictions": bm25_predictions,
            "hybrid_predictions": hybrid_predictions,
            "dense_recall": dense_metrics[0],
            "bm25_recall": bm25_metrics[0],
            "hybrid_recall": hybrid_metrics[0],
        },
        notes=["Dense, BM25, and hybrid retrieval comparison on a toy QA corpus."],
    )
    write_report(result, output_dir / "retrieval_demo.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
